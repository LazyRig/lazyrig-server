'use strict';
// Conversation state lives in IndexedDB. Inference state and results belong to
// the server; reconnecting never submits a second generation request.
window.Chat = (() => {
  const uuid=()=>Array.from(crypto.getRandomValues(new Uint8Array(16)),x=>x.toString(16).padStart(2,'0')).join('');
  const threads=new Map(), folders=new Map(), watching=new Map(), images=new Map();
  let db, folderDb, current, selectedFolder=null, ready=false, selecting=false, pendingSelection=null, readingFiles=0, files=[], attachmentVersion=0, modelSignature='', renderPending=false;
  const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const options=()=>$('#chat-options');
  const active=()=>threads.get(current);
  const pending=t=>t?.messages.findLast(m=>m.role==='assistant'&&!done(m.status));
  let storageNotice='Opening saved conversations…',storageWarning=false;
  const notice=text=>$('#chat-notice').textContent=storageWarning?'':storageNotice||text;
  const storageStatus=(text,warning=false,title='Chat cannot start yet')=>{storageNotice=text;storageWarning=warning;$('#chat-storage-warning').hidden=!warning;$('#chat-storage-warning-title').textContent=title;$('#chat-storage-warning-text').textContent=warning?text:'';notice('');};
  const auth=()=>key?{Authorization:'Bearer '+key}:{};
  // Opening the existing version avoids waiting for an old dashboard tab to
  // release its database. Folder metadata has its own database, so adding this
  // optional feature does not require upgrading the conversation database.
  function openDatabase(name,stores){return new Promise((resolve,reject)=>{
    const request=indexedDB.open(name);
    // An upgrade already queued by an older tab also blocks this versionless
    // open, but browsers do not emit onblocked for requests waiting behind it.
    const explainWait=()=>storageStatus('Close all other LazyRig tabs or windows, then reload this page. Chat cannot open until the history lock is released. Keep the same address and do not clear browser data.',true,'Chat blocked — close other LazyRig tabs');
    const timer=setTimeout(explainWait,3000);
    request.onupgradeneeded=()=>{for(const store of stores)if(!request.result.objectStoreNames.contains(store))request.result.createObjectStore(store,{keyPath:'id'});};
    request.onsuccess=()=>{clearTimeout(timer);const database=request.result;database.onversionchange=()=>{database.close();ready=false;storageStatus('Conversation storage changed in another tab. Reload this tab to continue.',true);updateButtons();};resolve(database);};
    request.onerror=()=>{clearTimeout(timer);reject(request.error);};request.onblocked=explainWait;
  });}
  function transaction(mode,operation,name='threads',database=name==='threads'?db:folderDb){return new Promise((resolve,reject)=>{const tx=database.transaction(name,mode),store=tx.objectStore(name);let result;const request=operation(store);request.onsuccess=()=>result=request.result;tx.oncomplete=()=>resolve(result);tx.onerror=tx.onabort=()=>reject(tx.error||new Error('Conversation storage failed'));});}
  async function importLegacyFolders(storedThreads){
    const legacy=db.objectStoreNames.contains('folders')?await transaction('readonly',store=>store.getAll(),'folders',db):[];
    await new Promise((resolve,reject)=>{const tx=folderDb.transaction(['folders','memberships','metadata'],'readwrite'),metadata=tx.objectStore('metadata'),read=metadata.get('legacy-import');
      read.onsuccess=()=>{if(read.result)return;for(const f of legacy)tx.objectStore('folders').put(f);for(const t of storedThreads)tx.objectStore('memberships').put({id:t.id,folderId:t.folderId||null});metadata.put({id:'legacy-import',complete:true});};
      tx.oncomplete=resolve;tx.onerror=tx.onabort=()=>reject(tx.error||new Error('Could not preserve existing folders'));
    });
  }
  async function deleteStoredThread(id){
    await transaction('readwrite',store=>store.delete(id));
    // A stale membership cannot resurrect a deleted conversation. Remove it
    // after the main record succeeds; keep deletion usable if cleanup fails.
    await transaction('readwrite',store=>store.delete(id),'memberships').catch(storageError);
  }
  // Saving a draft, a setting or streamed output does not change history order.
  // Recency belongs to a new conversation or an explicitly submitted user turn.
  async function save(t=active()){if(!t||!db)return;await transaction('readwrite',store=>store.put(structuredClone(t)));}
  async function initialize(){
    storageStatus('Opening saved conversations…');
    try{db=await openDatabase('lazyrig-chat',['threads']);folderDb=await openDatabase('lazyrig-chat-folders',['folders','memberships','metadata']);const storedThreads=await transaction('readonly',store=>store.getAll());await importLegacyFolders(storedThreads);for(const f of await transaction('readonly',store=>store.getAll(),'folders'))folders.set(f.id,f);const memberships=new Map((await transaction('readonly',store=>store.getAll(),'memberships')).map(m=>[m.id,m.folderId]));for(const t of storedThreads){try{const draft=JSON.parse(localStorage.getItem('lazyrig.draft.'+t.id)||'null');if(draft)Object.assign(t,draft);}catch{}if(memberships.has(t.id))t.folderId=memberships.get(t.id);threads.set(t.id,t);}ready=true;
      try{const savedFolder=localStorage.getItem('lazyrig.folder');selectedFolder=folders.has(savedFolder)?savedFolder:null;}catch{}
      storageStatus('');const saved=localStorage.getItem('lazyrig.chat')||[...threads.keys()][0];if(!threads.has(saved))await newChat();else select(saved);
      for(const t of threads.values())for(const m of t.messages)if(m.role==='assistant'&&!done(m.status)){
        if(m.workflow||m.jobId)watch(t,m);else{m.status='interrupted';m.error='Submission was not confirmed. Check Recent activity before sending again.';save(t).catch(storageError);}
      }
    }catch(e){ready=false;storageStatus('Conversation storage is unavailable: '+e.message,true);updateButtons();}
  }
  function storageError(e){notice('Could not save conversation: '+e.message);}
  function chatSettings(){return {...formObject(options()),enhance_prompt:options().elements.enhance_prompt.checked};}
  function defaultImageTool(){const available=state?.models.filter(m=>m.enabled!==false&&m.task==='text-to-image')||[];return available.find(m=>m.default_image_tool)||available[0];}
  function checkpoint(){const t=active();if(!t)return;t.draft=$('#chat-prompt').value;t.settings=chatSettings();try{localStorage.setItem('lazyrig.draft.'+t.id,JSON.stringify({draft:t.draft,settings:t.settings,task:t.task,model:t.model}));}catch(e){storageError(e);}}
  function newChatFolder(){return folders.get(selectedFolder)||null;}
  function updateNewChatDestination(){
    const folder=newChatFolder(),button=$('#chat-new');
    button.title=folder?`New chat in folder “${folder.name}”`:'New chat outside folders';
    button.setAttribute('aria-label',button.title);
    const compact=$('#chat-new-compact');compact.title=button.title;compact.setAttribute('aria-label',button.title);
    updateChatHeading();
  }
  async function newChat({title='New conversation',task=$('#chat-task').value||'chat',model=$('#chat-model').value,messages=[]}={}){
    $('#chat-history-search').value='';
    const folderId=newChatFolder()?.id||null;
    const t={id:uuid(),title,created:Date.now(),updated:Date.now(),folderId,task,model,settings:{},messages,draft:''};
    threads.set(t.id,t);select(t.id);await save(t);
    await transaction('readwrite',store=>store.put({id:t.id,folderId}),'memberships');$('#chat-prompt').focus();
    return t;
  }
  function select(id){
    // Model/size choices must exist before restoring saved select values.
    // IndexedDB and the first server-state request can finish in either order.
    if(!state){pendingSelection=id;return;}
    if(!threads.has(id))return;
    if(active())checkpoint();
    const initialDraft=!current?$('#chat-prompt').value:'';
    selecting=true;current=id;try{localStorage.setItem('lazyrig.chat',id);}catch(e){storageError(e);}attachmentVersion++;files=[];renderFiles();const t=active();
    selectThreadFolder(t);closeThreadMenu();clearHistoryDrag();
    if(!$$('#chat-threads [data-thread]').some(b=>b.dataset.thread===id))renderThreads();else updateThreadSelection();
    // A person can start typing before storage/server state finishes loading.
    if(initialDraft)t.draft=initialDraft;
    const desired=t.model;$('#chat-task').value=t.task;modelSignature='';updateModels();if(desired&&[...$('#chat-model').options].some(o=>o.value===desired))$('#chat-model').value=desired;
    options().reset();options().elements.image_model.value=defaultImageTool()?.id||'';for(const [name,value] of Object.entries(t.settings||{}))if(options().elements[name]){const el=options().elements[name];if(el.type==='checkbox')el.checked=value===true;else el.value=value;}
    $('#chat-prompt').value=t.draft||'';requestAnimationFrame(resizeComposer);updateTask();selecting=false;t.task=$('#chat-task').value;t.model=$('#chat-model').value;checkpoint();renderMessages(true);
    if(narrowChat.matches&&!$('#chat-history').hidden)setPanel('chat-history-toggle','chat-history','history',false,true);
  }
  const recentFirst=(a,b)=>(b.updated||b.created||0)-(a.updated||a.created||0)||(b.created||0)-(a.created||0)||String(a.id).localeCompare(String(b.id));
  const threadFolder=t=>folders.has(t.folderId)?t.folderId:null;
  // The displayed conversation can stay open while a different folder is selected.
  const threadSelected=t=>!!t&&t.id===current&&threadFolder(t)===selectedFolder;
  function selectFolder(id){
    selectedFolder=folders.has(id)?id:null;
    try{if(selectedFolder)localStorage.setItem('lazyrig.folder',selectedFolder);else localStorage.removeItem('lazyrig.folder');}catch(e){storageError(e);}
  }
  function selectThreadFolder(t){
    // The current selection controls both the highlight and New chat destination.
    // An outside conversation clears any previously selected folder.
    selectFolder(threadFolder(t));
  }
  const orderedFolders=()=>[...folders.values()].sort((a,b)=>a.name.localeCompare(b.name)||a.id.localeCompare(b.id));
  async function revealFolder(id){const f=folders.get(id);if(f?.collapsed){const next={...f,collapsed:false};await transaction('readwrite',store=>store.put(next),'folders');folders.set(id,next);}}
  async function moveThread(id,folderId){
    if(folderId&&!folders.has(folderId))throw new Error('That folder no longer exists.');
    const t=threads.get(id);if(!t)throw new Error('That conversation no longer exists.');
    if(threadFolder(t)===folderId)return;
    await transaction('readwrite',store=>store.put({id:t.id,folderId}),'memberships');t.folderId=folderId;if(t.id===current)selectThreadFolder(t);await save(t).catch(storageError);
    await revealFolder(folderId);
  }
  function renderThreads(){
    const historyTop=$('#chat-threads').scrollTop;
    updateNewChatDestination();
    closeThreadMenu();
    const items=folderId=>[...threads.values()].filter(t=>threadFolder(t)===folderId).sort(recentFirst).map(t=>`<div class="thread-row ${threadSelected(t)?'active':''}"><button class="thread-item ${threadSelected(t)?'active':''}" data-thread="${t.id}" draggable="${folders.size>0}" aria-current="${threadSelected(t)?'true':'false'}" title="${esc(t.title)}">${esc(t.title)}</button><button class="thread-menu-trigger icon-button" data-thread-menu="${t.id}" aria-label="Actions for ${esc(t.title)}" aria-haspopup="menu" aria-expanded="false" aria-controls="chat-thread-menu" title="Conversation actions">${UI.icon('more')}</button></div>`).join('');
    $('#chat-threads').innerHTML=orderedFolders().map(f=>{
      const count=[...threads.values()].filter(t=>threadFolder(t)===f.id).length;
      return `<div class="history-folder" data-folder="${f.id}" data-folder-drop="${f.id}"><div class="history-folder-row"><button class="history-folder-toggle" data-folder-toggle="${f.id}" aria-expanded="${!f.collapsed}" aria-current="${f.id===selectedFolder?'location':'false'}" aria-controls="folder-contents-${f.id}" title="${esc(f.name)}"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7V5h6l2 2h10v13H3Z"/>${f.collapsed?'':'<path d="M3 11h18"/>'}</svg><span>${esc(f.name)}</span>${count?`<small>${count}</small>`:''}</button><button class="history-folder-menu icon-button" data-folder-menu="${f.id}" aria-label="Manage folder ${esc(f.name)}" title="Manage folder">${UI.icon('more')}</button></div><div id="folder-contents-${f.id}" class="history-folder-contents" ${f.collapsed?'hidden':''}>${items(f.id)}</div></div>`;
    }).join('')+(folders.size?`<div class="history-unfiled" data-folder-drop="" aria-label="Conversations outside folders">${items(null)}</div>`:items(null));
    $('#chat-move').hidden=!folders.size||!active();
    for(const b of $$('#chat-threads [data-thread]'))b.onclick=()=>{if(b.dataset.thread!==current)select(b.dataset.thread);else{selectThreadFolder(active());updateThreadSelection();if(narrowChat.matches)setPanel('chat-history-toggle','chat-history','history',false,true);}};
    for(const b of $$('#chat-threads [data-thread-menu]'))b.onclick=()=>openThreadMenu(b);
    for(const b of $$('#chat-threads [data-folder-toggle]'))b.onclick=async()=>{
      const f=folders.get(b.dataset.folderToggle),next={...f,collapsed:!f.collapsed};
      selectFolder(f.id);folders.set(f.id,next);updateThreadSelection();
      try{await transaction('readwrite',store=>store.put(next),'folders');renderThreads();}
      catch(e){if(folders.get(f.id)===next){folders.set(f.id,f);renderThreads();}storageError(e);}
    };
    for(const b of $$('#chat-threads [data-folder-menu]'))b.onclick=()=>openHistoryDialog('edit',b.dataset.folderMenu);
    filterHistory();$('#chat-threads').scrollTop=historyTop;
  }
  function updateChatHeading(){
    const t=active(),folder=t&&folders.get(threadFolder(t));
    const title=$('#chat-title');title.textContent=t?.title||'New conversation';title.title=title.textContent;
    $('#chat-folder-label').hidden=!folder;$('#chat-workspace-label').hidden=!!folder;
    $('#chat-active-folder').textContent=folder?.name||'';$('#chat-active-folder').title=folder?.name||'';
  }
  function filterHistory(){
    const query=$('#chat-history-search').value.trim().toLocaleLowerCase();
    let visible=0;
    for(const row of $$('#chat-threads .thread-row')){
      const t=threads.get(row.querySelector('[data-thread]').dataset.thread),folder=t&&folders.get(threadFolder(t));
      row.hidden=!!query&&![t?.title,folder?.name].some(value=>String(value||'').toLocaleLowerCase().includes(query));
      if(!row.hidden)visible++;
    }
    for(const group of $$('#chat-threads .history-folder')){
      const f=folders.get(group.dataset.folder),hasVisible=!!group.querySelector('.thread-row:not([hidden])');
      group.hidden=!!query&&!hasVisible&&!f.name.toLocaleLowerCase().includes(query);
      const expanded=!!query||!f.collapsed;group.querySelector('.history-folder-contents').hidden=!expanded;
      group.querySelector('[data-folder-toggle]').setAttribute('aria-expanded',String(expanded));
    }
    let empty=$('#chat-history-empty');if(!empty){empty=document.createElement('div');empty.id='chat-history-empty';empty.className='history-empty';$('#chat-threads').append(empty);}
    empty.hidden=visible>0||!query;empty.textContent='No conversations found.';
    $('#chat-history-count').textContent=query?visible+' / '+threads.size:String(threads.size);
  }
  function updateThreadSelection(){
    updateNewChatDestination();
    for(const b of $$('#chat-threads [data-thread]')){const selected=threadSelected(threads.get(b.dataset.thread));b.classList.toggle('active',selected);b.setAttribute('aria-current',String(selected));b.closest('.thread-row').classList.toggle('active',selected);}
    for(const b of $$('#chat-threads [data-folder-toggle]'))b.setAttribute('aria-current',b.dataset.folderToggle===selectedFolder?'location':'false');
  }
  let draggedThreadId=null;
  const threadDragType='application/x-lazyrig-conversation';
  function clearHistoryDrag(){draggedThreadId=null;for(const el of $$('#chat-threads .is-drop-target,#chat-threads .is-dragging'))el.classList.remove('is-drop-target','is-dragging');}
  const historyList=$('#chat-threads');
  historyList.addEventListener('dragstart',e=>{
    const source=e.target.closest('[data-thread]');if(!source||!ready||!folders.size)return;
    draggedThreadId=source.dataset.thread;closeThreadMenu();e.dataTransfer.effectAllowed='move';e.dataTransfer.setData(threadDragType,draggedThreadId);source.closest('.thread-row').classList.add('is-dragging');
  });
  historyList.addEventListener('dragover',e=>{
    const target=e.target.closest('[data-folder-drop]');if(!draggedThreadId||!target)return;
    e.preventDefault();e.dataTransfer.dropEffect='move';for(const el of historyList.querySelectorAll('.is-drop-target'))if(el!==target)el.classList.remove('is-drop-target');target.classList.add('is-drop-target');
  });
  historyList.addEventListener('dragleave',e=>{const target=e.target.closest('[data-folder-drop]');if(target&&!target.contains(e.relatedTarget))target.classList.remove('is-drop-target');});
  historyList.addEventListener('drop',async e=>{
    const target=e.target.closest('[data-folder-drop]');if(!draggedThreadId||!target)return;
    e.preventDefault();const id=e.dataTransfer.getData(threadDragType),folderId=target.dataset.folderDrop||null,valid=id===draggedThreadId,overSelf=e.target.closest('[data-thread]')?.dataset.thread===id;clearHistoryDrag();if(!valid)return;
    // A few pixels of mouse movement can make the browser emit a drag instead
    // of a click. Releasing on the same conversation still selects it.
    if(overSelf){if(id!==current)select(id);return;}
    if(threadFolder(threads.get(id))===folderId)return;
    try{await moveThread(id,folderId);renderThreads();}catch(error){toast(error.message);}
  });
  document.addEventListener('dragend',clearHistoryDrag);
  window.addEventListener('blur',clearHistoryDrag);
  let threadMenuId=null,threadMenuButton=null;
  function closeThreadMenu(restoreFocus=false){
    $('#chat-thread-menu').hidden=true;
    const button=threadMenuButton;button?.setAttribute('aria-expanded','false');
    threadMenuButton=null;threadMenuId=null;
    if(restoreFocus&&button?.isConnected)button.focus({preventScroll:true});
  }
  function openThreadMenu(button){
    const id=button.dataset.threadMenu;if(threadMenuId===id){closeThreadMenu(true);return;}
    closeThreadMenu();const t=threads.get(id);if(!t)return;
    threadMenuId=id;threadMenuButton=button;button.setAttribute('aria-expanded','true');
    const menu=$('#chat-thread-menu');$('#chat-thread-menu-title').textContent=t.title;
    $('#chat-move').hidden=!folders.size;menu.hidden=false;
    const rect=button.getBoundingClientRect();
    menu.style.left=Math.max(8,Math.min(rect.left,innerWidth-menu.offsetWidth-8))+'px';
    menu.style.top=Math.max(8,Math.min(rect.bottom+4,innerHeight-menu.offsetHeight-8))+'px';
    menu.querySelector('button:not([hidden])').focus({preventScroll:true});
  }
  let historyDialogAction,historyDialogId;
  function openHistoryDialog(action,id=null){
    if(!ready)return;historyDialogAction=action;historyDialogId=id;
    const folder=folders.get(id),moving=action==='move',editing=action==='edit';
    const dialog=$('#chat-folder-dialog'),form=$('#chat-folder-form');form.reset();
    $('#chat-folder-title').textContent=moving?'Move conversation':editing?'Manage folder':'Add folder';
    $('#chat-folder-name-label').hidden=moving;form.elements.folder_name.required=!moving;form.elements.folder_name.value=editing?folder.name:'';
    $('#chat-folder-destination-label').hidden=!moving;
    form.elements.folder_id.innerHTML='<option value="">No folder</option>'+orderedFolders().map(f=>`<option value="${f.id}">${esc(f.name)}</option>`).join('');
    form.elements.folder_id.value=(moving?threadFolder(threads.get(id)):null)||'';
    $('#chat-folder-remove').hidden=!editing;$('#chat-folder-submit').textContent=moving?'Move':editing?'Save':'Add';
    $('#chat-folder-note').textContent=editing?'Removing a folder returns its conversations to the normal list.':'Folders organize conversations saved in this browser.';
    dialog.showModal();(moving?form.elements.folder_id:form.elements.folder_name).focus();
  }
  async function saveHistoryDialog(event){
    event.preventDefault();const form=event.target;
    try{
      if(historyDialogAction==='move'){
        await moveThread(historyDialogId,form.elements.folder_id.value||null);
      }else{
        const name=form.elements.folder_name.value.trim();if(!name)throw new Error('Enter a folder name.');
        const original=folders.get(historyDialogId),id=historyDialogAction==='edit'?historyDialogId:uuid();
        const next={id,name,collapsed:original?.collapsed||false};
        await transaction('readwrite',store=>store.put(next),'folders');folders.set(id,next);if(historyDialogAction==='create')selectFolder(id);
      }
      $('#chat-folder-dialog').close();renderThreads();
    }catch(e){toast(e.message);}
  }
  async function removeHistoryFolder(){
    const folder=folders.get(historyDialogId);if(!folder)return;
    if(!confirm(`Remove “${folder.name}”? Its conversations will stay in your history.`))return;
    const moved=[...threads.values()].filter(t=>threadFolder(t)===folder.id);
    try{
      await new Promise((resolve,reject)=>{const tx=folderDb.transaction(['folders','memberships'],'readwrite');tx.objectStore('folders').delete(folder.id);for(const t of moved)tx.objectStore('memberships').put({id:t.id,folderId:null});tx.oncomplete=resolve;tx.onerror=tx.onabort=()=>reject(tx.error||new Error('Could not remove folder'));});
      for(const t of moved){t.folderId=null;await save(t).catch(storageError);}
      folders.delete(folder.id);if(selectedFolder===folder.id)selectFolder(null);$('#chat-folder-dialog').close();renderThreads();
    }catch(e){storageError(e);}
  }
  function updateModels(preferredModel=''){
    if(!state)return;if(pendingSelection){const id=pendingSelection;pendingSelection=null;select(id);return;}const task=$('#chat-task').value;
    const available=state.models.filter(m=>m.enabled!==false&&(task==='chat'?m.task==='text-generation':task==='edit'?m.capabilities?.image_editing:m.task==='text-to-image'));
    const signature=JSON.stringify([task,available.map(m=>[m.id,m.generation,m.capabilities]),state.models.map(m=>[m.id,m.task,m.pipeline,m.enabled,m.capabilities?.image_editing])]);
    if(signature!==modelSignature){const old=preferredModel||active()?.model||$('#chat-model').value;$('#chat-model').innerHTML=available.map(m=>`<option value="${esc(m.id)}">${esc(m.id)}</option>`).join('')||'<option value="">No compatible models</option>';if(available.some(m=>m.id===old))$('#chat-model').value=old;modelSignature=signature;
      const im=options().elements.image_model,oldImage=im.value;im.innerHTML='<option value="">Disabled</option>'+modelOptions('text-to-image');im.value=oldImage;const enhancer=options().elements.enhancement_model,oldEnhancer=enhancer.value;enhancer.innerHTML=state.models.filter(m=>m.enabled!==false&&m.task==='text-generation'&&(task!=='edit'||['vlm','vlm-text'].includes(m.pipeline))).map(m=>`<option value="${esc(m.id)}">${esc(m.id)}</option>`).join('')||'<option value="">No compatible models</option>';if([...enhancer.options].some(o=>o.value===oldEnhancer))enhancer.value=oldEnhancer;updateTask();
    }
    updateModelState();updateButtons();
  }
  function updateModelState(){
    const model=selectedModelInfo(),hardware=[...(state.hardware?.devices||[]),...(state.scheduler.devices||[])];
    const modelState=model?(state.scheduler.states[model.id]||'unloaded'):'';
    const actual=model?.effective_placement?.devices||[model?.device];
    const target=actual.filter(Boolean).map(id=>{const name=hardware.find(d=>d.id===id)?.name;return name&&name!==id?id+' · '+name:id;}).join(' + ');
    $('#chat-model-state').textContent=model?[modelState,target].filter(Boolean).join(' · '):'';
    $('#chat-model-state').title=$('#chat-model-state').textContent;$('#chat-model').title=model?[model.id,$('#chat-model-state').textContent].filter(Boolean).join(' · '):'Select model';
    $('#chat-model-dot').dataset.state=modelState;
  }
  function selectedModelInfo(){return state?.models.find(m=>m.id===$('#chat-model').value);}
  function updateTask(){
    const task=$('#chat-task').value,model=selectedModelInfo(),vision=['vlm','vlm-text'].includes(model?.pipeline);
    if(active()&&active().task!==task)attachmentVersion++;
    $$('.chat-text-option').forEach(el=>el.hidden=task!=='chat');$$('.chat-image-option').forEach(el=>el.hidden=task==='chat');$$('.chat-strength-option').forEach(el=>el.hidden=task!=='edit'||!model?.image_editing?.supports_strength);
    const toolEnabled=task==='chat'&&!!options().elements.image_model.value;
    const toolButton=$('#chat-image-tool');toolButton.hidden=task!=='chat'||!state?.models.some(m=>m.task==='text-to-image'&&m.enabled!==false);toolButton.setAttribute('aria-pressed',String(toolEnabled));toolButton.innerHTML=UI.icon('image')+'<span>Image tool</span>';toolButton.title=toolEnabled?'Image tool: '+options().elements.image_model.value:'Enable image tool';toolButton.setAttribute('aria-label',toolButton.title);
    const imageModel=task==='chat'?state?.models.find(m=>m.id===options().elements.image_model.value):model;
    $$('.chat-size-option').forEach(el=>el.hidden=task==='chat'&&!toolEnabled);
    $$('.chat-enhancement-option').forEach(el=>el.hidden=task==='chat'&&!toolEnabled);$$('.chat-enhancement-model').forEach(el=>el.hidden=task==='chat'||!options().elements.enhance_prompt.checked);
    const attach=task==='edit'||(task==='chat'&&(vision||toolEnabled));$('#chat-attach').hidden=!attach;$('#chat-files').multiple=task!=='edit';
    $('#chat-prompt').placeholder=task==='image'?'Describe the image you want to create…':task==='edit'?'Describe how to edit your image…':vision?'Ask a question or attach images…':toolEnabled?'Chat or describe an image edit…':'Message your model…';
    const size=options().elements.size,old=size.value;size.innerHTML='<option value="auto">Automatic</option><option value="custom">Custom dimensions</option>'+(imageModel?.image_size_constraints?.presets||[]).filter(p=>p.width<=imageModel.max_width&&p.height<=imageModel.max_height).map(p=>`<option value="${p.width}x${p.height}">${p.aspect_ratio} · ${p.width} × ${p.height}</option>`).join('');if([...size.options].some(o=>o.value===old))size.value=old;
    for(const axis of ['width','height']){const input=options().elements[axis];input.min=imageModel?.image_size_constraints?.minimum||256;input.max=imageModel?.['max_'+axis]||1024;input.step=imageModel?.image_size_constraints?.multiple_of||64;}
    $$('.chat-custom-size').forEach(el=>el.hidden=(task==='chat'&&!toolEnabled)||size.value!=='custom');
    for(const name of ['max_tokens','temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty','steps','guidance_scale','seed','strength'])options().elements[name].placeholder=String(model?.generation?.[name]??(name==='max_tokens'?8192:'Model default'));
    options().elements.seed.min=task==='chat'?'0':'-1';
    if(!attach&&files.length){attachmentVersion++;files=[];renderFiles();}
    if(active()&&state&&!selecting){active().task=task;active().model=model?.id||'';checkpoint();save().catch(storageError);}window.Help.install();updateModelState();updateButtons();
  }
  function updateButtons(){const busy=!!pending(active());$('#chat-send').hidden=busy;$('#chat-stop').hidden=!busy;$('#chat-send').disabled=!ready||!$('#chat-model').value||readingFiles>0;$('#chat-stop').disabled=!!(pending(active())?.stopRequested||pending(active())?.cancelRequested);$('#chat-model').disabled=busy||readingFiles>0;$('#chat-task').disabled=busy||readingFiles>0;$('#chat-folder-new').disabled=!ready;$('#chat-new').disabled=!ready||!state;$('#chat-new-compact').disabled=$('#chat-new').disabled;}
  // Escape first, then add only a small set of known Markdown elements. Raw
  // HTML, remote images, scripts and arbitrary URL schemes are never rendered.
  function markdown(text){
    const blocks=String(text||'').split(/```/);return blocks.map((part,index)=>{
      if(index%2){const newline=part.indexOf('\n');return `<pre><button type="button" class="secondary code-copy" aria-label="Copy code" title="Copy code">${UI.icon('copy')}<span>Copy</span></button><code>${esc(newline>=0?part.slice(newline+1):part)}</code></pre>`;}
      let value=esc(part).replace(/`([^`\n]+)`/g,'<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g,'<strong>$1</strong>').replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
      return value.split(/\n\n+/).map(p=>/^#{1,3} /.test(p)?'<h3>'+p.replace(/^#{1,3} /,'')+'</h3>':'<p>'+p.replace(/\n/g,'<br>')+'</p>').join('');
    }).join('');
  }
  function bindCodeCopy(container){
    for(const button of container.querySelectorAll('.code-copy'))button.onclick=async()=>{
      if(!await copy(button.nextElementSibling.textContent))return;
      button.innerHTML=UI.icon('check')+'<span>Copied</span>';
      button.classList.add('copied');button.setAttribute('aria-label','Code copied');
      clearTimeout(button.copyReset);
      button.copyReset=setTimeout(()=>{button.innerHTML=UI.icon('copy')+'<span>Copy</span>';button.classList.remove('copied');button.setAttribute('aria-label','Copy code');},1800);
    };
  }
  function metricHTML(m){const c=m.context||{},t=m.timings||{},items=[];
    if(t.prompt_per_second!=null)items.push('Prompt ≈ '+Number(t.prompt_per_second).toFixed(1)+' t/s');
    if(t.predicted_per_second!=null)items.push('Decode '+Number(t.predicted_per_second).toFixed(1)+' t/s');
    if(c.context_limit)items.push('Context '+fmt(c.used_tokens||c.input_tokens)+' / '+fmt(c.context_limit)+(c.accuracy?.startsWith('estimated')?' ≈':''));
    if(m.finishReason==='length')items.push('Output limit reached');
    return items.map(v=>`<span>${esc(v)}</span>`).join('');
  }
  // Follow is user intent, not a distance recomputed on every token. A small
  // upward wheel/touch/key movement must win over the next streaming update.
  let followLatest=true, previousScrollTop=0, scrollingPointer=false, touchPosition=null;
  const atLatest=()=>{const el=$('#chat-messages');return el.scrollHeight-el.scrollTop-el.clientHeight<=2;};
  function readScrollPosition(){
    const top=$('#chat-messages').scrollTop;
    if(top<previousScrollTop-1)followLatest=false;
    else if(top>previousScrollTop+1&&atLatest()&&!scrollingPointer)followLatest=true;
    previousScrollTop=top;return top;
  }
  function syncChatScroll(force=false,savedTop=null){
    const el=$('#chat-messages');if(force)followLatest=true;
    if(followLatest&&!scrollingPointer)el.scrollTop=el.scrollHeight;
    else if(savedTop!==null)el.scrollTop=savedTop;
    previousScrollTop=el.scrollTop;$('#chat-latest').hidden=atLatest();
  }
  function imageRecoveryNotice(m){
    if(!m.imageRecovery)return '';
    const status=m.imageRecovery.status;
    const text=status==='succeeded'?'Recovered from a GPU resource error · retried once':status==='retrying'&&!done(m.status)?'GPU recovery · automatic retry 1/1':'Automatic GPU recovery did not complete (1/1)';
    return `<div class="fine image-recovery" data-recovery-status="${esc(status)}" role="status">${text}</div>`;
  }
  function renderMessages(scroll=false){
    const t=active();if(!t)return;const container=$('#chat-messages'),top=readScrollPosition();
    const opened=new Set([...container.querySelectorAll('details[open]')].map(e=>e.dataset.message));
    const savedImages=new Map([...container.querySelectorAll('article')].map(row=>[row.dataset.message,[...row.querySelector('.message-images').children]]));
    container.innerHTML=t.messages.length?t.messages.map(m=>`<article class="message ${m.role}" data-message="${m.id}"><div class="message-head"><strong>${m.role==='user'?'YOU':esc(m.model||'ASSISTANT')}</strong>${m.role==='assistant'?`<span>${esc(m.task==='chat'?'':m.task==='edit'?'IMAGE EDIT':'IMAGE GENERATION')}</span>`:''}</div><div class="message-card">${m.reasoning?`<details class="reasoning" data-message="${m.id}" ${opened.has(m.id)?'open':''}><summary>Reasoning</summary><div class="reasoning-text">${esc(m.reasoning)}</div></details>`:''}<div class="message-body">${m.role==='user'?esc(m.content).replace(/\n/g,'<br>'):markdown(m.content)}</div><div class="message-images">${(m.attachments||[]).map((a,index)=>`<img data-attachment-index="${index}" alt="${esc(a.name||'Input image')}">`).join('')}${(m.artifacts||[]).map(a=>`<figure data-artifact="${esc(a.url)}"><div class="fine">Loading saved image…</div></figure>`).join('')}</div>${m.role==='assistant'?`<div class="message-metrics">${metricHTML(m)}</div>${!done(m.status)?`<div class="job-progress"><span>${esc(m.stopRequested||m.cancelRequested?'Cancelling…':m.phase||'Submitting…')}</span>${m.progress?`<progress value="${m.progress.step}" max="${m.progress.steps}"></progress><span>${m.progress.step}/${m.progress.steps} steps</span>`:''}</div>`:''}${imageRecoveryNotice(m)}${m.enhancement?.rewritten_prompt?`<details class="image-prompt"><summary>Image prompt</summary><p>${esc(m.enhancement.rewritten_prompt)}</p><span class="fine">${m.enhancement.width} × ${m.enhancement.height}</span></details>`:''}${m.error?`<div class="job-error">${esc(m.error)}${m.status==='failed'?'<div class="fine">This failed request will not be resent with later messages.</div>':''}</div>`:''}<div class="message-tools">${UI.button('copy','Copy message',`data-copy-message="${m.id}"`)}${m.jobId?UI.button('code','Response JSON',`data-result="${m.jobId}"`)+UI.button('refresh','Refresh status',`data-refresh-job="${m.id}"`):''}</div>`:''}</div></article>`).join(''):
      '<div class="chat-empty"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 4H3v13h5v4l5-4h8Z"/><path d="M7 9h10M7 13h6"/></svg><h2>Start a conversation</h2><p>Chat, understand an image, or create one.</p></div>';
    // Keep decoded images and their occupied space across completion/status
    // rerenders. Loading placeholders can otherwise clamp the scroll position.
    const renderedMessages=new Map(t.messages.map(m=>[m.id,m]));
    for(const row of container.querySelectorAll('article')){
      const previous=savedImages.get(row.dataset.message)||[];
      for(const node of [...row.querySelector('.message-images').children]){
        const inputURL=node.tagName==='IMG'?renderedMessages.get(row.dataset.message)?.attachments?.[Number(node.dataset.attachmentIndex)]?.url:null;
        const index=previous.findIndex(old=>old.tagName===node.tagName&&(node.tagName==='IMG'?old.getAttribute('src')===inputURL:old.dataset.artifact===node.dataset.artifact&&old.querySelector('img')));
        if(index>=0)node.replaceWith(previous.splice(index,1)[0]);
        else if(inputURL){node.onload=()=>{if(node.isConnected)syncChatScroll();};node.src=inputURL;}
      }
    }
    bindCodeCopy(container);
    for(const b of container.querySelectorAll('[data-copy-message]'))b.onclick=()=>copy(t.messages.find(m=>m.id===b.dataset.copyMessage).content);
    for(const b of container.querySelectorAll('[data-result]'))b.onclick=async()=>{try{const m=t.messages.find(item=>item.jobId===b.dataset.result);downloadJSON(m?.workflow?{steps:m.workflow.completed,active_request:m.workflow.step?.id,status:m.status,error:m.error,image_recovery:m.imageRecovery,artifacts:m.artifacts,messages:m.apiHistory}:await api('/api/jobs/'+b.dataset.result+'/result'),'lazyrig-response-'+b.dataset.result+'.json');}catch(e){toast(e.message);}};
    for(const b of container.querySelectorAll('[data-refresh-job]'))b.onclick=()=>{const m=t.messages.find(m=>m.id===b.dataset.refreshJob);if(m)watch(t,m);};
    for(const figure of container.querySelectorAll('[data-artifact]'))if(!figure.querySelector('img'))loadArtifact(figure).catch(e=>{if(figure.isConnected)figure.textContent=e.message;});
    syncChatScroll(scroll,top);updateButtons();
    notice(pending(t)?'Accepted requests continue on the server. This browser resumes the tool conversation when you return.':'Conversations are saved in this browser. Generated files are saved on the server.');
  }
  function scheduleRender(t){
    if(t.id!==current||renderPending)return;renderPending=true;
    requestAnimationFrame(()=>{
      renderPending=false;if(t.id!==current)return;
      const m=t.messages.findLast(item=>item.role==='assistant');
      const row=m&&document.querySelector(`article[data-message="${m.id}"]`);
      if(!row||done(m.status)||(m.artifacts||[]).length!==row.querySelectorAll('figure').length||
          (m.imageRecovery?.status||'')!==(row.querySelector('.image-recovery')?.dataset.recoveryStatus||'')){renderMessages();return;}
      const container=$('#chat-messages'),top=readScrollPosition();
      row.querySelector('.message-body').innerHTML=markdown(m.content);
      let reasoning=row.querySelector('.reasoning');
      if(m.reasoning&&!reasoning){reasoning=document.createElement('details');reasoning.className='reasoning';reasoning.dataset.message=m.id;reasoning.innerHTML='<summary>Reasoning</summary><div class="reasoning-text"></div>';row.querySelector('.message-body').before(reasoning);}
      if(reasoning)reasoning.querySelector('.reasoning-text').textContent=m.reasoning||'';
      row.querySelector('.message-metrics').innerHTML=metricHTML(m);
      const progress=row.querySelector('.job-progress');if(progress)progress.innerHTML=`<span>${esc(m.stopRequested||m.cancelRequested?'Cancelling…':m.phase||'Working…')}</span>${m.progress?`<progress value="${m.progress.step}" max="${m.progress.steps}"></progress><span>${m.progress.step}/${m.progress.steps} steps</span>`:''}`;
      bindCodeCopy(row);
      syncChatScroll(false,top);updateButtons();
    });
  }
  async function artifactBlob(url){
    const parsed=new URL(url,location.origin);if(parsed.origin!==location.origin||!/^\/artifacts\/[a-f0-9]{32}\.(png|svg)$/.test(parsed.pathname))throw new Error('Invalid artifact URL');
    if(!images.has(parsed.pathname))images.set(parsed.pathname,(async()=>{const res=await fetch(parsed.pathname,{headers:auth()});if(res.status!==200)throw new Error('Image unavailable (HTTP '+res.status+'). Refresh status to retry.');const blob=await res.blob();return {blob,url:URL.createObjectURL(blob),path:parsed.pathname};})().catch(e=>{images.delete(parsed.pathname);throw e;}));
    return images.get(parsed.pathname);
  }
  async function loadArtifact(figure){const asset=await artifactBlob(figure.dataset.artifact);if(!figure.isConnected)return;
    figure.innerHTML=`<img src="${asset.url}" alt="Generated image"><figcaption><a href="${asset.url}" download="${asset.path.split('/').pop()}">${UI.icon('import')}Download</a><button class="artifact-use">${UI.icon('chat')}Use in chat</button><button class="artifact-edit">${UI.icon('image')}Edit image</button><button class="artifact-url" aria-label="Copy image URL" title="Copy image URL">${UI.icon('copy')}</button></figcaption>`;
    figure.querySelector('img').onload=()=>{if(figure.isConnected)syncChatScroll();};
    figure.querySelector('.artifact-url').onclick=()=>copy(location.origin+asset.path);
    figure.querySelector('.artifact-edit').onclick=()=>useArtifact(asset,'edit').catch(e=>toast(e.message));figure.querySelector('.artifact-use').onclick=()=>useArtifact(asset,'chat').catch(e=>toast(e.message));
  }
  async function useArtifact(asset,task){
    if(pending(active()))throw new Error('Wait for this job to finish before changing tasks.');
    const enabled=state.models.filter(m=>m.enabled!==false),selected=$('#chat-model').value;
    const editors=enabled.filter(m=>m.capabilities?.image_editing);
    const imageModel=editors.find(m=>m.id===(task==='edit'?selected:options().elements.image_model.value))||editors.find(m=>m.id===selected)||editors.find(m=>m.default_image_tool)||editors[0];
    const chats=enabled.filter(m=>m.task==='text-generation'),vision=chats.filter(m=>['vlm','vlm-text'].includes(m.pipeline));
    const compatible=task==='edit'?imageModel:vision.find(m=>m.id===selected)||vision[0]||(imageModel&&(chats.find(m=>m.id===selected)||chats[0]));
    if(!compatible)throw new Error(task==='edit'?'Register an image-edit model first.':'Register a vision-language model, or a chat model and an image-edit model first.');
    // Set the tool before rebuilding model options so a text-only chat with an
    // image tool keeps its attachments while the task changes.
    if(task==='chat'&&imageModel)options().elements.image_model.value=imageModel.id;
    $('#chat-task').value=task;modelSignature='';updateModels(compatible.id);
    if(await addFiles([new File([asset.blob],'generated.png',{type:asset.blob.type})],asset.path))$('#chat-prompt').focus();
  }
  async function addFiles(incoming,artifactPath=null){
    if($('#chat-attach').hidden)throw new Error('Select a vision model, enable the chat Image tool, or choose Edit image to attach a source.');
    if($('#chat-task').value==='edit'&&incoming.length>1)throw new Error('Image editing accepts one source image per request. Choose one image; Chat & vision can analyze multiple images.');
    const task=$('#chat-task').value;
    if(task==='edit')attachmentVersion++;
    const version=attachmentVersion;
    const attached=artifactPath&&files.find(f=>f.artifactPath===artifactPath);
    if(attached){if(task==='edit'){files=[attached];renderFiles();}return true;}
    readingFiles++;updateButtons();try {
    const incomingFiles=[];
    for(const file of incoming){if(!/^image\/(png|jpeg|webp|gif)$/.test(file.type))throw new Error('Use PNG, JPEG, WebP or GIF.');if(file.size>10*1024*1024)throw new Error('Each image must be at most 10 MiB.');const url=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('Could not read image'));reader.readAsDataURL(file);});incomingFiles.push({name:file.name,url,bytes:file.size,...(artifactPath?{artifactPath}:{})});}
    if(version!==attachmentVersion||task!==$('#chat-task').value||$('#chat-attach').hidden)return false;
    // Merge with the current composer after reading, so overlapping clicks
    // cannot append duplicates or overwrite a different image added meanwhile.
    const next=task==='edit'?[]:[...files];
    for(const file of incomingFiles){const index=next.findIndex(f=>(file.artifactPath&&f.artifactPath===file.artifactPath)||f.url===file.url);if(index<0)next.push(file);else if(file.artifactPath)next[index]={...next[index],artifactPath:file.artifactPath};}
    if(next.reduce((n,f)=>n+f.bytes,0)>24*1024*1024)throw new Error('Attachments exceed 24 MiB.');files=next;renderFiles();return true;
    }finally{readingFiles--;updateButtons();}
  }
  function renderFiles(){$('#chat-attachments').innerHTML=files.map((f,i)=>`<div class="attachment"><img src="${esc(f.url)}" alt="${esc(f.name)}"><button type="button" data-remove-file="${i}" aria-label="Remove ${esc(f.name)}">×</button></div>`).join('');for(const b of $$('[data-remove-file]'))b.onclick=()=>{files.splice(Number(b.dataset.removeFile),1);renderFiles();};}
  function requestHistory(t){
    // Keep failed turns on screen, but do not replay their input or attachments.
    // Derive this from saved replies so existing conversations recover as well.
    return t.messages.filter((message,index)=>{
      const reply=t.messages[index+1];
      return message.role!=='user'||reply?.role!=='assistant'||reply.status!=='failed';
    });
  }
  function imageSources(t){
    const sources=[];
    for(const m of requestHistory(t)){
      for(const a of m.attachments||[])sources.push({url:a.url,description:('Uploaded '+(a.name||'image')+' with request: '+m.content).slice(0,512)});
      for(const a of m.artifacts||[])sources.push({url:new URL(a.url,location.origin).pathname,description:('Previously '+(a.operation==='edit'?'edited':'generated')+' image: '+(a.prompt||m.content)).slice(0,512)});
    }
    return sources.map((source,index)=>({...source,id:'image_'+(index+1)}));
  }
  function requestBody(t,user){
    const settings=chatSettings(),body={model:t.model},history=requestHistory(t);
    if(t.task==='chat'){
      const vision=['vlm','vlm-text'].includes(selectedModelInfo()?.pipeline);
      if(!vision&&!settings.image_model&&history.some(m=>m.task==='chat'&&m.attachments?.length))throw new Error('This conversation contains images. Select a vision model, enable the Image tool for edits, or start a new chat.');
      body.messages=[];if(settings.system_prompt.trim())body.messages.push({role:'system',content:settings.system_prompt});
      for(const m of history){if(m.role==='user'&&m.task==='chat')body.messages.push({role:'user',content:vision&&m.attachments?.length?[{type:'text',text:m.content},...m.attachments.map(a=>({type:'image_url',image_url:{url:a.url}}))]:m.content});else if(m.role==='assistant'&&m.task==='chat'&&m.status==='completed')body.messages.push(...(m.apiHistory||[m.apiMessage||{role:'assistant',content:m.content}]).map(message=>!vision&&Array.isArray(message.content)?{...message,content:message.content.filter(p=>p.type==='text').map(p=>p.text).join('\n')}:message));}
      for(const name of ['max_tokens','temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty','seed'])if(settings[name]!=='')body[name]=Number(settings[name]);
      body.timings_per_token=true;
      if(settings.image_model){const imageModel=state.models.find(m=>m.id===settings.image_model);body.tools=InferenceClient.imageTools(imageModel,imageSources(t),settings.enhance_prompt,InferenceClient.settingsSize(imageModel,settings));}
      if(settings.thinking!=='')body.chat_template_kwargs={enable_thinking:settings.thinking==='true'};
    }else{
      body.prompt=user.content;body.size=InferenceClient.settingsSize(selectedModelInfo(),settings);body.response_format='url';
      for(const name of ['steps','guidance_scale','seed'])if(settings[name]!=='')body[name]=Number(settings[name]);
      if(settings.negative_prompt!=='')body.negative_prompt=settings.negative_prompt;
      if(t.task==='edit'){if(user.attachments.length!==1)throw new Error('Image editing accepts one source image per request. Attach one image or remove the extras.');body.image=user.attachments[0].url;if(settings.strength!==''&&selectedModelInfo()?.image_editing?.supports_strength)body.strength=Number(settings.strength);}
      if(settings.enhance_prompt&&!settings.enhancement_model)throw new Error(t.task==='edit'?'Select a vision-language model for edit prompt enhancement.':'Select an enhancement model.');
    }
    return body;
  }
  async function send(event){
    event.preventDefault();const t=active();if(!t||pending(t)||readingFiles)return;const prompt=$('#chat-prompt').value.trim();if(!prompt)return;
    t.model=$('#chat-model').value;t.task=$('#chat-task').value;t.settings=chatSettings();const user={id:uuid(),role:'user',task:t.task,content:prompt,attachments:structuredClone(files)};t.messages.push(user);
    let body;try{body=requestBody(t,user);}catch(e){t.messages.pop();toast(e.message);return;}
    const message={id:uuid(),role:'assistant',model:t.model,task:t.task,status:'submitting',content:'',artifacts:[]};
    try{message.workflow=InferenceClient.create({task:t.task,body,settings:t.settings,models:state.models,sources:imageSources(t),user});}catch(e){t.messages.pop();toast(e.message);return;}
    t.messages.push(message);if(t.title==='New conversation')t.title=prompt.slice(0,50);
    const previousUpdated=t.updated;t.updated=Date.now();
    try{await save(t);}catch(e){t.updated=previousUpdated;t.messages.splice(-2);storageError(e);return;} // Persist the receipt key before any network mutation.
    // Move focus before Send is hidden. Chromium can otherwise adjust the
    // message scroll position and make the adjustment look like user scrolling.
    $('#chat-prompt').focus({preventScroll:true});
    $('#chat-prompt').value='';resizeComposer();t.draft='';checkpoint();files=[];renderFiles();renderThreads();renderMessages(true);
    watch(t,message);
  }
  function applyJob(m,job){m.cancelRequested=!!job.cancel_requested;m.status=job.status;m.phase=job.phase;m.progress=job.progress;m.content=job.content??job.text??m.content;m.reasoning=job.reasoning_content||'';m.artifacts=job.artifacts||[];m.context=job.context;m.timings=job.timings;m.error=job.error;m.enhancement=job.prompt_enhancement;m.imageRecovery=job.image_recovery;}
  async function cancelJob(t,m){
    if(!m.jobId||done(m.status)||m.cancelRequested)return;
    const result=await api('/api/jobs/'+m.jobId+'/cancel',{});
    m.cancelRequested=!!result.cancel_requested;m.phase='Cancelling';await save(t);scheduleRender(t);
  }
  async function finish(t,m){
    const job=await api('/api/jobs/'+m.jobId);applyJob(m,job);
    // Stop intent is saved before HTTP. Retry only the idempotent cancel call
    // after receipt/reconnect, never the generation submission.
    if(m.stopRequested&&!done(m.status))await cancelJob(t,m);
    if(job.status==='completed'){
      try{const result=await api('/api/jobs/'+m.jobId+'/result');const choice=result.choices?.[0];if(choice){m.apiMessage=choice.message;m.content=choice.message.content||'';m.reasoning=choice.message.reasoning_content||m.reasoning;m.finishReason=choice.finish_reason;}m.artifacts=result.x_artifacts||m.artifacts;m.context=result.x_context||m.context;m.timings=result.timings||m.timings;}catch(e){m.error='Job saved; response fetch failed: '+e.message;}
    }
    await save(t);scheduleRender(t);return done(m.status);
  }
  async function watch(t,m){
    if(!m.workflow)return watchLegacy(t,m);
    if(watching.has(m.id)||done(m.status))return;
    watching.set(m.id,true);
    try{await InferenceClient.run(t,m,{save,render:scheduleRender,controller:value=>watching.set(m.id,value),
      isStopped:()=>localStorage.getItem('lazyrig.stop.'+m.id)==='true',
      fetch:(url,opts={})=>fetch(url,{...opts,headers:{'Content-Type':'application/json',...auth(),...opts.headers}}),
      source:async url=>{
        let data=url;if(!url.startsWith('data:')){const asset=await artifactBlob(url);data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(reader.error);reader.readAsDataURL(asset.blob);});}
        const img=new Image();img.src=data;await img.decode();return {url:data,width:img.naturalWidth,height:img.naturalHeight};
      }});}catch(e){m.status='failed';m.error=e.message;await save(t).catch(storageError);scheduleRender(t);}
    finally{watching.delete(m.id);updateButtons();}
  }
  async function watchLegacy(t,m){
    if(watching.has(m.jobId))return;watching.set(m.jobId,true);let delay=1000;
    try{for(;;){
      try{
        if(await finish(t,m))break;
        const controller=new AbortController();watching.set(m.jobId,controller);let watchdog=setTimeout(()=>controller.abort(),35000);
        const response=await fetch('/api/jobs/'+m.jobId+'/events?after='+(m.seq||0),{headers:auth(),signal:controller.signal}).catch(e=>{clearTimeout(watchdog);throw e;});if(!response.ok){clearTimeout(watchdog);const e=new Error('Progress HTTP '+response.status);e.status=response.status;throw e;}
        const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',checkpoint=Date.now();
        try{for(;;){const {value,done:ended}=await reader.read();clearTimeout(watchdog);watchdog=setTimeout(()=>controller.abort(),35000);if(ended)break;buffer+=decoder.decode(value,{stream:true});let end;
          while((end=buffer.indexOf('\n\n'))>=0){const block=buffer.slice(0,end);buffer=buffer.slice(end+2);const raw=block.split('\n').find(line=>line.startsWith('data: '));if(!raw||block.includes('event: end'))continue;const e=JSON.parse(raw.slice(6));m.seq=e.seq||m.seq;
            if(e.kind==='text'){m.content=e.content??e.text;if(t.settings.image_model&&e.stage!=='continuation')m.content=m.content.split('<tool_call')[0];m.reasoning=e.reasoning_content;m.context=e.context;m.timings=e.timings||m.timings;m.phase='Generating response';}
            else if(e.kind==='enhancement_started'||e.kind==='enhancement_progress')m.phase='Enhancing image prompt';
            else if(e.kind==='prompt_enhanced'){m.enhancement=e.enhancement;m.phase='Prompt ready · preparing image model';}
            else if(e.kind==='image_retry'){m.progress=null;m.imageRecovery=e.recovery;m.phase=e.message;}
            else if(e.kind==='image_recovered')m.imageRecovery=e.recovery;
            else if(e.kind==='progress'){m.progress={step:e.step,steps:e.steps};m.phase='Denoising';}
            else if(e.kind==='context')m.context=e;
            else if(e.kind==='artifact'){if(!m.artifacts.some(a=>a.url===e.artifact.url))m.artifacts.push(e.artifact);m.phase='Image saved · completing handoff';}
            else if(['loading','loaded','unloading','unloaded','restoring','resuming','image_started','tool_call'].includes(e.kind))m.phase=e.kind.replaceAll('_',' ')+' '+(e.model||'');
            scheduleRender(t);
          }
          if(Date.now()-checkpoint>1000){await save(t);checkpoint=Date.now();}
        }}finally{clearTimeout(watchdog);reader.cancel().catch(()=>{});reader.releaseLock();}
        delay=1000;
      }catch(e){
        if(e.status===401){m.phase='API key required · reconnect to resume';scheduleRender(t);break;}
        if(e.status===404){m.status='interrupted';m.error='No saved job found for this receipt. Check Recent activity before trying again.';await save(t);scheduleRender(t);break;}
        m.phase='Connection lost · reconnecting to saved job';scheduleRender(t);await wait(delay);delay=Math.min(delay*2,10000);
      }
    }}finally{watching.delete(m.jobId);updateButtons();}
  }
  let recoveryQueue=Promise.resolve();
  function recover(jid){
    // Serialize activity opens so repeated clicks cannot both create a missing
    // conversation. Wait for saved history before deciding whether it is new.
    recoveryQueue=recoveryQueue.then(async()=>{
      await initialized;if(!ready)return;
      // A tool conversation owns every completed stage, not just its latest
      // receipt. Prefer the oldest owner if an earlier app made a recovery copy.
      for(const t of [...threads.values()].sort((a,b)=>a.created-b.created)){
        const m=t.messages.find(m=>m.role==='assistant'&&(m.jobId===jid||m.workflow?.step?.id===jid||m.workflow?.completed?.some(step=>step.id===jid)));
        if(!m)continue;
        await revealFolder(threadFolder(t));renderThreads();select(t.id);
        if(!done(m.status))watch(t,m);return;
      }
      const job=await api('/api/jobs/'+jid),task=job.kind==='image'?(job.operation==='edit'?'edit':'image'):'chat';
      const m={id:uuid(),role:'assistant',model:job.model,task,jobId:jid,content:'',artifacts:[]};applyJob(m,job);
      const t=await newChat({title:(job.prompt||'Recovered job').slice(0,50),task,model:job.model,
        messages:[{id:uuid(),role:'user',content:job.prompt||'Recovered request',task,attachments:[]},m]});
      watch(t,m);
    }).catch(e=>toast(e.message));
    return recoveryQueue;
  }
  function downloadJSON(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
  $('#chat-composer').onsubmit=send;
  $('#chat-new').onclick=$('#chat-new-compact').onclick=()=>newChat().catch(storageError);
  $('#chat-history-search').oninput=()=>{closeThreadMenu();filterHistory();$('#chat-threads').scrollTop=0;};
  $('#chat-folder-new').onclick=()=>openHistoryDialog('create');
  $('#chat-move').onclick=()=>{const id=threadMenuId;closeThreadMenu();if(threads.has(id))openHistoryDialog('move',id);};
  $('#chat-folder-form').onsubmit=saveHistoryDialog;
  $('#chat-folder-remove').onclick=()=>removeHistoryFolder();
  $('#chat-task').onchange=()=>{modelSignature='';updateModels();updateTask();};$('#chat-model').onchange=updateTask;
  options().oninput=()=>{checkpoint();save().catch(storageError);};
  options().onchange=()=>{updateTask();checkpoint();if(active()){active().settings=chatSettings();save().catch(storageError);}};
  options().onreset=()=>{if(!selecting)setTimeout(()=>{updateTask();if(active()){active().settings=chatSettings();save().catch(storageError);}},0);};
  $('#chat-attach').onclick=()=>$('#chat-files').click();$('#chat-files').onchange=async e=>{try{await addFiles(e.target.files);}catch(error){toast(error.message);}e.target.value='';};
  function resizeComposer(){
    const field=$('#chat-prompt'),container=$('#chat-messages');
    if(!field.clientWidth)return;
    const top=readScrollPosition();
    field.style.height='auto';field.style.height=Math.min(field.scrollHeight,160,innerHeight*.22)+'px';
    syncChatScroll(false,top);
  }
  window.addEventListener('resize',()=>requestAnimationFrame(resizeComposer));
  $('#chat-prompt').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();if(!pending(active()))$('#chat-composer').requestSubmit();}};
  let draftTimer;$('#chat-prompt').oninput=()=>{resizeComposer();checkpoint();const t=active();clearTimeout(draftTimer);draftTimer=setTimeout(()=>{if(t&&threads.get(t.id)===t)save(t).catch(storageError);},500);};
  $('#chat-prompt').onpaste=e=>{const incoming=[...(e.clipboardData?.files||[])];if(incoming.length){e.preventDefault();addFiles(incoming).catch(error=>toast(error.message));}};
  const composer=$('#chat-composer');composer.ondragover=e=>{if(![...e.dataTransfer.types].includes('Files'))return;e.preventDefault();composer.classList.add('drop-target');};composer.ondragleave=()=>composer.classList.remove('drop-target');composer.ondrop=e=>{composer.classList.remove('drop-target');if(!e.dataTransfer.files.length)return;e.preventDefault();addFiles(e.dataTransfer.files).catch(error=>toast(error.message));};
  $('#chat-stop').onclick=async()=>{
    const t=active(),m=pending(t);if(!m)return;
    m.stopRequested=true;m.phase='Cancelling';scheduleRender(t);
    try{localStorage.setItem('lazyrig.stop.'+m.id,'true');await save(t);if(m.status!=='submitting')await cancelJob(t,m);}
    catch(e){watching.get(m.workflow?m.id:m.jobId)?.abort?.();toast('Stop request not confirmed: '+e.message+'. Reconnecting will retry cancellation.');}
  };
  const narrowChat=matchMedia('(max-width:900px)');
  const panels=[['chat-history-toggle','chat-history','history'],['chat-settings-toggle','chat-options','parameters']];
  function setPanel(button,panel,label,open,restoreFocus=false){
    const container=$('#chat-messages'),top=readScrollPosition();
    closeThreadMenu();
    window.Help.hide();$('#'+panel).hidden=!open;
    if(panel==='chat-options'){if(open)UI.open($('#chat-settings-dialog'));else UI.close($('#chat-settings-dialog'));}
    $('#'+button).setAttribute('aria-expanded',String(open));$('#'+button).title=panel==='chat-options'?'Conversation settings':(open?'Hide ':'Show ')+label;
    if(panel==='chat-history'){$('#'+button).setAttribute('aria-label',(open?'Hide ':'Show ')+'history');updateHistoryLayout();}
    window.Help.install();if(open&&panel==='chat-options')options().scrollTop=0;
    if(restoreFocus)$('#'+button).focus({preventScroll:true});
    syncChatScroll(false,top);
    if(open&&panel==='chat-history'&&narrowChat.matches)$('#chat-history-search').focus({preventScroll:true});
  }
  function updateHistoryLayout(){
    const open=!$('#chat-history').hidden,mobile=narrowChat.matches;
    $('#chat-history-scrim').hidden=!open||!mobile;
    $('#chat-new-compact').hidden=open;
    $('.chat-workspace').inert=open&&mobile;
  }
  $('#chat-history-close').onclick=$('#chat-history-scrim').onclick=()=>setPanel('chat-history-toggle','chat-history','history',false,true);
  narrowChat.addEventListener('change',()=>setPanel('chat-history-toggle','chat-history','history',!narrowChat.matches));
  for(const [button,panel,label] of panels)$('#'+button).onclick=()=>setPanel(button,panel,label,$('#'+panel).hidden);
  $('#close-chat-settings').onclick=$('#done-chat-settings').onclick=()=>setPanel('chat-settings-toggle','chat-options','parameters',false,true);
  $('#chat-settings-dialog').addEventListener('close',()=>{options().hidden=true;$('#chat-settings-toggle').setAttribute('aria-expanded','false');$('#chat-settings-toggle').title='Conversation settings';window.Help.hide();});
  let previousImageTool='';
  $('#chat-image-tool').onclick=()=>{const field=options().elements.image_model;if(field.value){previousImageTool=field.value;field.value='';}else{field.value=state.models.find(m=>m.id===previousImageTool&&m.task==='text-to-image'&&m.enabled!==false)?.id||defaultImageTool()?.id||'';}field.dispatchEvent(new Event('change',{bubbles:true}));};
  document.addEventListener('keydown',e=>{
    if(e.key==='Escape'&&!$('#chat-thread-menu').hidden){e.preventDefault();closeThreadMenu(true);return;}
    if(e.key!=='Escape'||e.defaultPrevented||$('#chat-view').hidden||document.querySelector('dialog[open]'))return;
    const open=panels.filter(([,panel])=>!$('#'+panel).hidden);
    const target=open.find(([button,panel])=>$('#'+panel).contains(e.target)||e.target===$('#'+button))||open.find(([,panel])=>panel==='chat-options')||open[0];
    if(target){e.preventDefault();setPanel(...target,false,true);}
  });
  $('#chat-latest').onclick=()=>syncChatScroll(true);
  const messageList=$('#chat-messages');messageList.tabIndex=0;
  messageList.addEventListener('wheel',e=>{if(e.deltaY<0)followLatest=false;},{passive:true});
  messageList.addEventListener('touchstart',e=>{touchPosition=e.touches[0]?.clientY;},{passive:true});
  messageList.addEventListener('touchmove',e=>{const y=e.touches[0]?.clientY;if(y>touchPosition)followLatest=false;touchPosition=y;},{passive:true});
  messageList.addEventListener('keydown',e=>{
    if(['ArrowUp','PageUp','Home'].includes(e.key)||(e.key===' '&&e.shiftKey))followLatest=false;
  });
  messageList.addEventListener('pointerdown',()=>{scrollingPointer=true;followLatest=false;});
  const releaseScroll=()=>{if(!scrollingPointer)return;scrollingPointer=false;followLatest=atLatest();syncChatScroll();};
  window.addEventListener('pointerup',releaseScroll);window.addEventListener('pointercancel',releaseScroll);
  messageList.onscroll=()=>{
    readScrollPosition();$('#chat-latest').hidden=atLatest();
  };
  $('#chat-delete').onclick=async()=>{
    const t=threads.get(threadMenuId);closeThreadMenu();if(!t)return;
    if(pending(t)){toast('Stop or finish the active job before deleting this conversation.');return;}
    if(!confirm('Delete “'+t.title+'”? Server artifacts remain available.'))return;
    try{
      await deleteStoredThread(t.id);localStorage.removeItem('lazyrig.draft.'+t.id);threads.delete(t.id);
      if(current===t.id){
        // Keep the old ID until selection so its draft is not treated as a startup draft.
        const next=[...threads.values()].sort(recentFirst)[0];if(next)select(next.id);else await newChat();
      }
      renderThreads();
    }catch(e){storageError(e);}
  };
  $('#chat-export').onclick=()=>{const t=threads.get(threadMenuId);closeThreadMenu();if(t)downloadJSON(t,'lazyrig-conversation.json');};
  document.addEventListener('pointerdown',e=>{if(!e.target.closest('#chat-thread-menu,[data-thread-menu]'))closeThreadMenu();});
  $('#chat-threads').addEventListener('scroll',()=>closeThreadMenu(),{passive:true});
  window.addEventListener('resize',()=>closeThreadMenu());
  $('#chat-thread-menu').addEventListener('keydown',e=>{
    if(e.key==='Tab'){closeThreadMenu(true);return;}
    if(!['ArrowDown','ArrowUp','Home','End'].includes(e.key))return;
    e.preventDefault();const buttons=[...e.currentTarget.querySelectorAll('button:not([hidden])')],index=buttons.indexOf(document.activeElement);
    const next=e.key==='Home'?0:e.key==='End'?buttons.length-1:(index+(e.key==='ArrowDown'?1:-1)+buttons.length)%buttons.length;
    buttons[next].focus();
  });
  window.addEventListener('pagehide',()=>{checkpoint();save().catch(()=>{});});
  document.addEventListener('keydown',e=>{
    if(e.key!=='Tab'||!narrowChat.matches||$('#chat-history').hidden||$('#chat-view').hidden||document.querySelector('dialog[open]'))return;
    const controls=[...$('#chat-history').querySelectorAll('button,input,[tabindex="0"]')].filter(el=>!el.disabled&&el.getClientRects().length),first=controls[0],last=controls.at(-1);
    if(e.shiftKey&&(document.activeElement===first||!$('#chat-history').contains(document.activeElement))){e.preventDefault();last?.focus();}
    else if(!e.shiftKey&&(document.activeElement===last||!$('#chat-history').contains(document.activeElement))){e.preventDefault();first?.focus();}
  });
  setPanel('chat-history-toggle','chat-history','history',!narrowChat.matches);
  updateButtons();
  const initialized=initialize();
  return {show(){for(const t of threads.values())for(const m of t.messages)if(m.role==='assistant'&&(m.workflow||m.jobId)&&!done(m.status))watch(t,m);updateModels();renderThreads();renderMessages(true);requestAnimationFrame(resizeComposer);},updateModels,recover,layout:resizeComposer};
})();
