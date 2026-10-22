'use strict';
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=n=>Number(n||0).toLocaleString(), done=s=>['completed','failed','cancelled','interrupted'].includes(s);
const fmtMemory=n=>Number(n||0).toLocaleString(undefined,{minimumFractionDigits:1,maximumFractionDigits:1});
const views=['models','devices','server','downloads','chat'];
let runtimeUpdateState=null, runtimeUpdatePending=sessionStorage.getItem('lazyrig.runtime-update')||'', runtimeUpdateTimer, runtimeUpdatePolling=false;
let modelFilter='all';
let state, view='models', key=sessionStorage.getItem('lazyrig.key')||'', selectedModel, toastTimer, toastKind='', folderTarget, folderState, pendingRefresh=false;
// Keep refresh in this tab's current section; chat restores its own conversation.
try{const savedView=sessionStorage.getItem('lazyrig.view');if(views.includes(savedView))view=savedView;}catch{}
function hideToast(){clearTimeout(toastTimer);const notice=$('#toast');notice.hidden=true;toastKind='';document.body.append(notice);}
function toast(s,kind=''){
  const notice=$('#toast');
  const dialog=document.activeElement?.closest('dialog[open]')||$$('dialog[open]').at(-1);
  // A modal's backdrop covers page-level notices regardless of their z-index.
  // Keep the alert inside the active dialog so it remains readable and accessible.
  if(dialog){const heading=dialog.querySelector('.panel-heading, h2');if(heading)heading.after(notice);else dialog.prepend(notice);}
  else document.body.append(notice);
  toastKind=kind;notice.textContent=s;notice.hidden=false;
  clearTimeout(toastTimer);toastTimer=setTimeout(hideToast,7000);
}
document.addEventListener('close',event=>{if(event.target.tagName==='DIALOG'&&!event.target.open&&event.target.contains($('#toast')))hideToast();},true);
async function api(path,body,timeout=0){const response=await fetch(path,{...(timeout?{signal:AbortSignal.timeout(timeout)}:{}),method:body===undefined?'GET':'POST',headers:{...(body!==undefined?{'Content-Type':'application/json'}:{}),...(key?{Authorization:'Bearer '+key}:{})},...(body!==undefined?{body:JSON.stringify(body)}:{})});const data=await response.json();if(!response.ok){if(response.status===401&&!$('#key-dialog').open)$('#key-dialog').showModal();const error=new Error(data.error?.message||data.error||response.statusText);error.data=data;error.status=response.status;throw error;}return data;}
function formObject(form){return Object.fromEntries(new FormData(form));}
function numbers(o,keys){for(const k of keys)if(k in o)o[k]=Number(o[k]);return o;}
function status(s){return `<span class="status ${esc(s)}">${esc(s)}</span>`;}
function deviceLabel(d){return `${d.id} · ${d.name}${d.support_level==='experimental'?' · Experimental':''}${d.detected===false?' · Not detected':''}`;}
function preferredDevice(){
  const devices=placementDevices().filter(d=>d.detected!==false),recommended=state?.hardware.recommended_device;
  if(devices.some(d=>d.id===recommended))return recommended;
  const gpus=devices.filter(d=>d.id.startsWith('GPU')&&d.support_level==='supported').sort((a,b)=>Number(b.device_type==='discrete')-Number(a.device_type==='discrete')||Number(b.budget_gib||0)-Number(a.budget_gib||0));
  return (gpus[0]||devices.find(d=>d.id==='CPU')||devices[0])?.id;
}
function deviceOptions(value){if(!placementDevices().some(d=>d.id===value))value=preferredDevice();return placementDevices().map(d=>`<option value="${esc(d.id)}" ${value===d.id?'selected':''}>${esc(deviceLabel(d))}</option>`).join('');}
function mtpDeviceOptions(value=''){return `<option value="" ${!value?'selected':''}>Same as model</option>`+state.scheduler.devices.filter(d=>/^(CPU|GPU)(\.\d+)*$/.test(d.id)).map(d=>`<option value="${esc(d.id)}" ${value===d.id?'selected':''}>${esc(deviceLabel(d))}</option>`).join('');}
function placementDevices(){return state?.scheduler.devices||[];}
function placementName(id){const d=placementDevices().find(d=>d.id===id);return d?deviceLabel(d):id;}
function placementAllowsMultiple(m){return m.task==='text-generation'&&!m.placement_info?.custom_properties&&m.placement_info?.multi_device_supported!==false;}
function normalizedPlacement(m,value=m.placement){
  const mode=['auto','single','split'].includes(value?.mode)?value.mode:'auto';
  const known=new Set(placementDevices().map(d=>d.id));
  let devices=[...new Set(value?.devices||m.placement_info?.default_devices||[m.device])].filter(id=>known.has(id));
  if(!devices.length&&known.has(m.device))devices=[m.device];
  if(!devices.length&&placementDevices().length)devices=[placementDevices()[0].id];
  if(!placementAllowsMultiple(m)||mode==='single')devices=devices.slice(0,1);
  if(devices.length>1){
    devices=devices.filter(id=>!id.startsWith('NPU'));
    if(devices[0]==='CPU')devices=[...devices.filter(id=>id!=='CPU'),'CPU'];
    else devices=[devices[0],...devices.slice(1).filter(id=>id!=='CPU'),...devices.slice(1).filter(id=>id==='CPU')];
  }
  return {mode:placementAllowsMultiple(m)?mode:'single',devices,shares:{},allow_fallback:mode==='auto'&&value?.allow_fallback===true,shared_memory:mode!=='split'&&m.task==='text-generation'&&devices[0]?.startsWith('GPU')&&value?.shared_memory==='allow'?'allow':'budget'};
}
function placementPayload(value){return {...value,devices:[...value.devices],shares:{},allow_fallback:value.mode==='auto'&&value.allow_fallback,shared_memory:value.mode==='split'?'budget':value.shared_memory};}
function placementCaption(value){return value.devices.length?({auto:'Auto',single:'Single',split:'Split'}[value.mode]+' · '+value.devices.join(' + ')):'Choose devices';}
function placementProblem(m,value,mtp=m.mtp?.enabled){
  if(!value.devices.length)return 'Select at least one device.';
  if(value.mode==='split'){
    if(mtp)return 'Split mode is unavailable while MTP is enabled. Disable MTP explicitly, or choose Automatic / Single device.';
    if(value.devices.length<2)return 'Select at least two devices for Split mode.';
    const host=value.devices.filter(id=>id==='CPU'||placementDevices().find(d=>d.id===id)?.device_type==='integrated');
    if(host.length>1)return 'Select either an integrated GPU or CPU as the shared-memory stage. Automatic can try them as alternatives.';
  }
  return '';
}
function placementSummary(value,mtp){
  if(value.mode==='split')return 'Split across all selected devices. No whole-model fallback. OpenVINO decides the partition; device budgets do not set partition sizes.';
  let note=value.mode==='single'?'Only the primary device is used.':'Try the primary device first. '+(mtp?'Splitting is disabled by MTP.':value.devices.length>1?'Then try a supported split across selected devices.':'Select more devices to allow a split.');
  if(value.shared_memory==='allow')note+=' If stored weights exceed its budget, permit a primary GPU shared-memory attempt'+(value.mode==='auto'?' after splitting.':'.');
  if(value.mode==='auto')note+=value.allow_fallback?' Finally, allow the whole model on another selected device.':' Whole-model fallback is off.';
  return note;
}
function devicePicker(m,value){
  return `<details class="device-picker" data-device-picker><summary title="${esc(value.devices.map(placementName).join(' + '))}" aria-label="Devices for ${esc(m.id)}"><span data-device-caption>${esc(placementCaption(value))}</span></summary><div class="device-picker-menu">
    <label>Execution mode<select data-placement-mode aria-label="Execution mode"><option value="auto">Automatic</option><option value="single">Single device</option><option value="split">Split across selected devices</option></select></label>
    <label>Primary device<select data-placement-primary aria-label="Primary device for ${esc(m.id)}"></select></label>
    <fieldset class="device-picker-choices"><legend>Devices</legend>${placementDevices().map(d=>`<label class="checkbox"><input type="checkbox" value="${esc(d.id)}" data-placement-device><span>${esc(placementName(d.id))}</span></label>`).join('')}</fieldset>
    <label class="checkbox placement-policy"><input type="checkbox" data-placement-fallback><span>Allow whole-model fallback to another selected device</span></label>
    <label class="checkbox placement-policy"><input type="checkbox" data-placement-shared><span>Allow primary GPU shared-memory attempt above its planning budget</span></label>
    <p class="fine placement-shared-help">The driver decides whether system RAM can be used. This permits an attempt; it does not allocate extra VRAM. System RAM guards remain active.</p>
    <p class="fine device-picker-note" data-placement-note role="status"></p>
    <div class="actions"><button type="button" class="primary" data-placement-apply>Apply</button><button type="button" class="secondary" data-placement-cancel>Cancel</button></div>
  </div></details>`;
}
function bindDevicePicker(container,m,initial,onApply){
  const picker=container.querySelector('[data-device-picker]'),primary=picker.querySelector('[data-placement-primary]'),choices=[...picker.querySelectorAll('[data-placement-device]')],apply=picker.querySelector('[data-placement-apply]'),note=picker.querySelector('[data-placement-note]'),mode=picker.querySelector('[data-placement-mode]'),fallback=picker.querySelector('[data-placement-fallback]'),shared=picker.querySelector('[data-placement-shared]');
  const mtpEnabled=()=>container.closest('form')?.elements.mtp_enabled?.value==='true'||(!container.closest('form')?.elements.mtp_enabled&&m.mtp?.enabled);
  let saved=normalizedPlacement(m,initial),draft=structuredClone(saved),busy=false,errorMessage='';
  const position=()=>{
    if(!picker.open)return;
    const panel=picker.querySelector('.device-picker-menu'),rect=picker.querySelector('summary').getBoundingClientRect(),width=Math.min(370,innerWidth-24);
    panel.style.width=width+'px';panel.style.left=Math.max(12,Math.min(rect.left,innerWidth-width-12))+'px';
    const height=Math.min(panel.scrollHeight,innerHeight-24),below=rect.bottom+6;
    panel.style.top=(below+height<=innerHeight-12?below:Math.max(12,rect.top-height-6))+'px';
  };
  const render=()=>{
    const multiple=placementAllowsMultiple(m)&&draft.mode!=='single',npu=draft.devices.some(id=>id.startsWith('NPU')),custom=!!m.placement_info?.custom_properties;
    const problem=placementProblem(m,draft,mtpEnabled());
    mode.value=draft.mode;mode.disabled=busy||custom||!placementAllowsMultiple(m);
    mode.querySelector('[value="split"]').disabled=!!mtpEnabled()||!placementAllowsMultiple(m);
    fallback.checked=draft.allow_fallback;fallback.disabled=busy||custom||draft.mode!=='auto';
    shared.checked=draft.shared_memory==='allow';shared.disabled=busy||custom||m.task!=='text-generation'||draft.mode==='split'||!draft.devices[0]?.startsWith('GPU');
    for(const choice of choices){choice.checked=draft.devices.includes(choice.value);choice.disabled=busy||custom;}
    const candidates=placementDevices().map(device=>device.id),gpuSelected=multiple&&draft.devices.some(id=>id.startsWith('GPU'));
    primary.innerHTML=(!draft.devices.length?'<option value="" disabled selected>Choose primary device</option>':'')+candidates.map(id=>`<option value="${esc(id)}" ${draft.devices[0]===id?'selected':''} ${id==='CPU'&&gpuSelected?'disabled title="Use CPU alone to make it primary."':''}>${esc(placementName(id))}</option>`).join('');
    primary.disabled=busy||custom||!candidates.length;
    apply.disabled=busy||custom||!!problem;
    picker.querySelector('[data-placement-cancel]').disabled=busy;
    const experimental=draft.devices.some(id=>placementDevices().find(d=>d.id===id)?.support_level==='experimental');
    const allocationNote=custom?(m.placement_info.reason||'Custom runtime properties control placement. Remove those properties and save before using these controls.'):problem||placementSummary(draft,mtpEnabled());
    note.textContent=errorMessage||[allocationNote,experimental?'Experimental devices remain selectable; compatibility is not validated.':''].filter(Boolean).join(' ');
    position();
  };
  const updateCaption=()=>{const summary=picker.querySelector('summary'),names=saved.devices.map(placementName).join(' + ');picker.querySelector('[data-device-caption]').textContent=placementCaption(saved);summary.title=names;summary.setAttribute('aria-label',`Devices for ${m.id}: ${names}`);};
  picker.querySelector('summary').addEventListener('click',event=>{
    event.preventDefault();if(busy)return;picker.open=!picker.open;
    if(picker.open){for(const other of document.querySelectorAll('[data-device-picker][open]'))if(other!==picker)other.open=false;draft=structuredClone(saved);errorMessage='';render();}
  });
  // Native toggle is queued: it must not reset a draft after a quick change.
  picker.addEventListener('toggle',()=>{if(picker.open)position();});
  for(const choice of choices)choice.addEventListener('change',()=>{
    errorMessage='';
    if(choice.checked){
      if(!placementAllowsMultiple(m)||draft.mode==='single'||choice.value.startsWith('NPU')||draft.devices.some(id=>id.startsWith('NPU')))draft.devices=[choice.value];
      else if(!draft.devices.includes(choice.value))draft.devices.push(choice.value);
    }else draft.devices=draft.devices.filter(id=>id!==choice.value);
    if(draft.devices.length)draft=normalizedPlacement(m,draft);
    render();
  });
  mode.onchange=()=>{errorMessage='';draft=normalizedPlacement(m,{...draft,mode:mode.value});render();};
  fallback.onchange=()=>{draft.allow_fallback=fallback.checked;render();};
  shared.onchange=()=>{draft.shared_memory=shared.checked?'allow':'budget';render();};
  container.closest('form')?.elements.mtp_enabled?.addEventListener('change',render);
  primary.addEventListener('change',()=>{
    errorMessage='';
    const selected=primary.value;
    const devices=!placementAllowsMultiple(m)||draft.mode==='single'||selected.startsWith('NPU')?[selected]:[selected,...draft.devices.filter(id=>id!==selected&&!id.startsWith('NPU'))];
    draft=normalizedPlacement(m,{...draft,devices});render();
  });
  picker.querySelector('[data-placement-cancel]').onclick=()=>{picker.open=false;draft=structuredClone(saved);render();};
  picker.addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();event.stopPropagation();picker.open=false;picker.querySelector('summary').focus();}});
  apply.onclick=async()=>{
    if(busy||m.placement_info?.custom_properties||placementProblem(m,draft,mtpEnabled()))return;
    busy=true;render();
    try{await onApply(structuredClone(draft));saved=structuredClone(draft);updateCaption();picker.open=false;}
    catch(error){errorMessage=error.message;}
    finally{busy=false;render();}
  };
  updateCaption();render();
  return {get:()=>structuredClone(saved),set:value=>{saved=normalizedPlacement(m,value);draft=structuredClone(saved);updateCaption();if(picker.open)render();}};
}
function placementEditor(m){
  const value=normalizedPlacement(m),effective=m.effective_placement||m.placement_info?.effective;
  return `<div class="model-placement" data-placement-editor><div class="placement-heading"><span class="field-label">Devices</span>${devicePicker(m,value)}</div><input type="hidden" name="device" value="${esc(value.devices[0]||m.device)}">
    <p class="fine placement-effective" data-placement-effective ${effective?'':'hidden'} title="${esc(effective?.target||'')}">${effective?`Currently loaded: ${esc((effective.devices||[]).map(placementName).join(' + ')||effective.label||effective.target)}`:''}</p>
    <p class="fine placement-auto" data-placement-auto></p><p class="fine placement-warning" data-reservation-warning></p>
  </div>`;
}
function syncPlacementEffective(){
  const element=$('#model-form [data-placement-effective]'),model=state?.models.find(m=>m.id===selectedModel);
  if(!element||!model)return;
  const effective=model.effective_placement||model.placement_info?.effective;
  element.hidden=!effective;
  element.textContent=effective?`Currently loaded: ${(effective.devices||[]).map(placementName).join(' + ')||effective.label||effective.target}`:'';
  element.title=effective?[effective.target,...(effective.notes||[])].filter(Boolean).join('\n'):'';
}
function bindPlacementEditor(form,m){
  const container=form.querySelector('[data-placement-editor]');
  let value=normalizedPlacement(m);
  const render=()=>{
    const mtp=form.elements.mtp_enabled?.value==='true';
    container.querySelector('[data-placement-auto]').textContent=placementProblem(m,value,mtp)||placementSummary(value,mtp);
    const weights=m.placement_info?.reservation_floor_gib,warning=container.querySelector('[data-reservation-warning]');
    warning.textContent=value.mode!=='split'&&weights!=null&&weights>Number(form.elements.memory_gib.value)?`The reservation is below the stored language weights (${fmtMemory(weights)} GiB). Full-GPU loads reserve at least this amount; cache and buffers need extra headroom.`:'';
    warning.hidden=!warning.textContent;
  };
  bindDevicePicker(container,m,value,next=>{value=normalizedPlacement(m,next);const previous=form.elements.device.value;form.elements.device.value=value.devices[0];if(previous!==value.devices[0])form.elements.device.dispatchEvent(new Event('change'));render();});
  render();
  form.elements.memory_gib.addEventListener('input',render);form.elements.mtp_enabled?.addEventListener('change',render);
  return ()=>{if(m.placement_info?.custom_properties)return undefined;const problem=placementProblem(m,value,form.elements.mtp_enabled?.value==='true');if(problem)throw new Error(problem);return placementPayload(value);};
}
document.addEventListener('click',event=>{for(const picker of document.querySelectorAll('[data-device-picker][open]'))if(!picker.contains(event.target))picker.open=false;});
window.addEventListener('resize',()=>{for(const picker of document.querySelectorAll('[data-device-picker][open]'))picker.open=false;});
document.addEventListener('scroll',event=>{for(const picker of document.querySelectorAll('[data-device-picker][open]'))if(!picker.contains(event.target))picker.open=false;},true);
function updateMemoryDefault(form, apply=false){
  const device=state?.scheduler.devices.find(d=>d.id===form.elements.device.value);
  const input=form.elements.memory_gib, button=form.querySelector('[data-memory-default]');
  const budget=Number(device?.budget_gib), valid=Number.isFinite(budget)&&budget>0;
  if(apply)input.value=valid?String(budget):'';
  button.disabled=!valid;
  button.title=valid?`Use ${device.id} device budget (${fmtMemory(budget)} GiB)`:'Select a configured device first';
}
function bindMemoryDefault(form, apply=false){
  form.elements.device.onchange=()=>updateMemoryDefault(form,true);
  form.querySelector('[data-memory-default]').onclick=()=>updateMemoryDefault(form,true);
  updateMemoryDefault(form,apply);
}
function modelOptions(task){return state.models.filter(m=>m.task===task&&m.enabled!==false).map(m=>`<option value="${esc(m.id)}">${esc(m.id)}</option>`).join('');}
function show(name){
  if(!views.includes(name))name='models';
  window.Help?.hide();view=name;$('#view-label').textContent={models:'Models',devices:'Devices',server:'API & server',downloads:'Imports',chat:'Chat'}[name];
  try{sessionStorage.setItem('lazyrig.view',name);}catch{}
  for(const v of views)$(`#${v}-view`).hidden=v!==name;
  $$('nav [data-view]').forEach(b=>{b.classList.toggle('active',b.dataset.view===name);b.setAttribute('aria-current',b.dataset.view===name?'page':'false');});
  if(name==='models')renderModels();
  if(name==='devices')renderDevices();
  if(name==='server')renderServer();
  if(name==='downloads')renderDownloads();
  document.body.classList.toggle('chat-open',name==='chat');
  if(name==='chat')window.Chat?.show();
  $('#server-activity').hidden=name!=='models';
  window.Help?.install();
}
async function refresh(full=false){if(pendingRefresh)return;pendingRefresh=true;try{state=await api('/api/state?summary=1');if(toastKind==='authentication')hideToast();$('#connection').textContent='Server online';const versionLabel='v'+state.version.replace(/beta(-test)?$/, ' beta$1');$('#version').textContent=versionLabel;document.title='LazyRig · '+versionLabel;$('#backend-label').textContent='OpenVINO™';$('#backend-version').textContent=state.hardware.openvino_version||state.hardware.note;$('#base-url').textContent=location.origin+'/v1';$('#registered-count').textContent=state.models.length;$('#loaded-count').textContent=Object.values(state.scheduler.states).filter(s=>s==='loaded').length;$('#job-count').textContent=state.jobs.filter(j=>!done(j.status)&&j.status!=='queued').length+' / '+state.jobs.filter(j=>j.status==='queued').length;$('#runtime-banner').hidden=state.hardware.available;$('#runtime-banner').textContent=state.hardware.note;if(full)show(view);else{$$('[data-model-status]').forEach(el=>el.innerHTML=status(state.scheduler.states[el.dataset.modelStatus]||'unloaded'));if(view==='downloads')renderDownloads();if(view==='server')renderConnections();}syncLoadControls();syncPlacementEffective();renderActivity();window.Chat?.updateModels();window.Help?.install();}catch(e){$('#connection').textContent=runtimeUpdatePending?'Updating runtime…':'Disconnected';if(full)toast(e.message,e.status===401?'authentication':'');}finally{pendingRefresh=false;}}
function renderModels(){
  $('#detected-devices').textContent='Detected hardware: '+(state.hardware.devices.map(deviceLabel).join(' | ')||'none');
  $('#models-empty').hidden=!!state.models.length;
  $('#models-table').innerHTML=state.models.map(m=>`<tr data-model-row="${esc(m.id)}" data-task="${esc(m.task)}"><td><div class="model-identity"><span class="model-avatar ${m.task==='text-to-image'?'image-avatar':''}">${UI.icon(m.task==='text-to-image'?'image':'spark')}</span><div><button class="model-name" data-edit="${esc(m.id)}">${esc(m.name||m.id)}</button><small class="model-alias">${esc(m.id)}</small><div class="model-meta">${m.task==='text-generation'?(['vlm','vlm-text'].includes(m.pipeline)?'Chat & vision':'Text generation'):'Image generation'}${m.enabled===false?' · API disabled':''}${m.default_image_tool?`<span class="default-tool" title="Default image tool">${UI.icon('image')} Default tool</span>`:''}</div></div></div></td><td data-model-device="${esc(m.id)}">${devicePicker(m,normalizedPlacement(m))}</td><td data-model-status="${esc(m.id)}">${status(state.scheduler.states[m.id]||'unloaded')}</td><td><div class="controls">${UI.button('play','Load '+m.id,`data-load="${esc(m.id)}"`,'load-model')}${UI.button('stop','Unload '+m.id,`data-unload="${esc(m.id)}"`,'unload-model')}${UI.button('folder','Inspect model folder',`data-inspect-model="${esc(m.id)}"`)}${UI.button('gear','Settings for '+m.id,`data-edit="${esc(m.id)}"`)}</div></td></tr>`).join('');
  $$('[data-edit]').forEach(b=>b.onclick=()=>editModel(b.dataset.edit));
  $$('[data-load]').forEach(b=>b.onclick=()=>control(b.dataset.load,'load'));
  $$('[data-unload]').forEach(b=>b.onclick=()=>control(b.dataset.unload,'unload'));
  $$('[data-inspect-model]').forEach(b=>b.onclick=()=>inspectModel(b.dataset.inspectModel));
  $$('[data-model-device]').forEach(cell=>{
    const model=state.models.find(m=>m.id===cell.dataset.modelDevice);
    bindDevicePicker(cell,model,normalizedPlacement(model),async value=>{
      await api('/api/settings',{model:model.id,device:value.devices[0],placement:placementPayload(value)});
      toast('Devices saved. Click Load or send a request to apply them.');
      await refresh(true);
      if(selectedModel===model.id&&$('#model-editor').open)editModel(selectedModel);
    });
  });
  if(selectedModel&&!state.models.some(m=>m.id===selectedModel))UI.close($('#model-editor'));
  filterModels();syncLoadControls();syncModelEditorTriggers();
}
async function inspectModel(id){
  const model=state.models.find(m=>m.id===id);if(!model)return;
  const dialog=$('#model-inspection-dialog');dialog.dataset.model=id;
  dialog.innerHTML=`<div class="modal-heading"><div><h2 id="inspection-title">Model folder</h2><p class="fine">${esc(model.name||id)}</p></div>${UI.button('x','Close model inspection','data-modal-close')}</div><div class="settings-scroll"><p class="fine inspection-path">${esc(model.path)}</p><div id="model-inspection-content" role="status">Inspecting exported model…</div></div><div class="settings-footer"><button type="button" class="primary" data-modal-close>Done</button></div>`;
  UI.open(dialog);
  try{
    const result=await api('/api/models/inspect',{path:model.path});
    if(!dialog.open||dialog.dataset.model!==id)return;
    $('#model-inspection-content').innerHTML=`<div class="grid two"><div><div class="eyebrow">ARCHITECTURE</div><p>${esc(result.architecture)}</p></div><div><div class="eyebrow">PIPELINE</div><p>${esc(result.pipeline)}</p></div><div><div class="eyebrow">IR COMPONENTS</div><p>${esc(result.ir_components)}</p></div><div><div class="eyebrow">SIZE ON DISK</div><p>${fmtMemory(Number(result.size_bytes||0)/1024**3)} GiB</p></div></div><p class="fine">${esc(result.validation)}</p>`;
  }catch(error){if(dialog.open&&dialog.dataset.model===id){$('#model-inspection-content').textContent=error.message;}}
}
function filterModels(){
  const query=$('#model-search').value.trim().toLowerCase();let visible=0;
  for(const row of $$('[data-model-row]')){row.hidden=(modelFilter!=='all'&&row.dataset.task!==modelFilter)||(query&&!row.querySelector('.model-identity').textContent.toLowerCase().includes(query));if(!row.hidden)visible++;}
  $('#model-no-matches').hidden=!state?.models.length||visible>0;
  $$('[data-model-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.modelFilter===modelFilter)));
}
function syncLoadControls(){
  for(const row of $$('[data-model-row]')){
    const current=state.scheduler.states[row.dataset.modelRow]||'unloaded',loaded=current==='loaded',busy=['loading','unloading'].includes(current);
    const load=row.querySelector('[data-load]'),unload=row.querySelector('[data-unload]');
    load.hidden=loaded;unload.hidden=!loaded;load.disabled=busy;unload.disabled=busy;
  }
}
async function control(id,action){try{const j=await api(`/api/models/${encodeURIComponent(id)}/${action}`,{});toast(`${action==='load'?'Loading':'Unloading'} ${id} · job ${j.id.slice(0,8)}`);await refresh();}catch(e){toast(e.message);}}
function runtimeControl(c,m){
  const value=(m[c.scope]||{})[c.key], name='runtime_'+c.key;
  const cpuThreads=c.key==='INFERENCE_NUM_THREADS',defaultLabel=cpuThreads?'Device default':c.default_label||'Runtime default';
  const attrs=`name="${name}" data-runtime-key="${esc(c.key)}" data-runtime-scope="${c.scope}" data-runtime-kind="${c.kind}"`;
  let input;
  if(c.kind==='select'||c.kind==='boolean'){
    const options=c.kind==='boolean'?[['true','Enabled'],['false','Disabled']]:c.options;
    const selected=value===undefined?'':String(value);
    const choices=[['','Runtime default'],...options];
    if(!choices.some(([v])=>v===selected))choices.push([selected,selected+' · custom']);
    input=`<select ${attrs}>${choices.map(([v,label])=>`<option value="${esc(v)}" ${v===selected?'selected':''}>${esc(label)}</option>`).join('')}</select>`;
  }else{
    input=`<input ${attrs} type="${c.kind==='integer'?'number':'text'}" value="${esc(value??'')}" placeholder="${esc(defaultLabel)}" ${c.kind==='integer'?`min="${c.minimum??0}" max="2147483647" step="1"`:''}>`;
  }
  const help=cpuThreads?'Blank uses the CPU inference threads setting under Devices. A value here overrides that default for this model.':c.help||'';
  return `<label data-help="${esc(help)}">${esc(c.label)}${input}</label>`;
}
function runtimeSection(section,m){return (state.runtime_controls||[]).filter(c=>c.section===section).map(c=>runtimeControl(c,m)).join('');}
function bindRuntimeControls(form){
  const object=field=>{const value=JSON.parse(field.value);if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Settings must be a JSON object');return value;};
  const controls=[...form.querySelectorAll('[data-runtime-key]')];
  for(const scope of ['plugin_config','scheduler_config']){
    const field=form.elements[scope];if(!field)continue;
    field.oninput=()=>{
      try{
        const values=object(field);field.setCustomValidity('');
        for(const input of controls.filter(c=>c.dataset.runtimeScope===scope)){
          const v=values[input.dataset.runtimeKey];
          if(input.tagName==='SELECT'&&v!==undefined&&![...input.options].some(o=>o.value===String(v)))input.add(new Option(String(v)+' · custom',String(v)));
          input.value=v===undefined?'':String(v);input.setCustomValidity('');
        }
      }catch(e){field.setCustomValidity(e.message);}
    };
  }
  for(const input of controls){
    input.oninput=input.onchange=()=>{
      const field=form.elements[input.dataset.runtimeScope];
      try{
        const values=object(field),value=input.value.trim(),kind=input.dataset.runtimeKind,k=input.dataset.runtimeKey;
        if(value==='')delete values[k];
        else values[k]=kind==='integer'?Number(value):kind==='boolean'?value==='true':kind==='streams'&&/^\d+$/.test(value)?Number(value):value;
        field.value=JSON.stringify(values,null,2);field.setCustomValidity('');input.setCustomValidity('');
      }catch(e){input.setCustomValidity('Fix the advanced '+input.dataset.runtimeScope+' JSON first: '+e.message);}
    };
  }
}
function imageSchedulerSection(m){
  const info=m.diffusion_scheduler_info||{},saved=m.diffusion_scheduler||{},overrides=saved.config||{};
  const controls=(info.controls||[]).map(c=>{
    const value=overrides[c.key],fallback=info.export_config?.[c.key],attributes=`data-diffusion-key="${esc(c.key)}" data-kind="${c.kind}"`;
    const input=c.kind==='boolean'?`<select ${attributes}><option value="">Export default${fallback===undefined?'':` · ${fallback}`}</option><option value="true" ${value===true?'selected':''}>Enabled</option><option value="false" ${value===false?'selected':''}>Disabled</option></select>`:`<input ${attributes} type="number" step="${c.kind==='integer'?'1':'any'}" value="${esc(value??'')}" placeholder="Export default${fallback===undefined?'':` · ${fallback}`}">`;
    return `<label>${esc(c.label)}${input}</label>`;
  }).join('');
  return `<h2>Diffusion scheduler</h2>
    <label>Sampler<select name="image_scheduler_type">${(info.types||['AUTO']).map(type=>`<option value="${type}" ${(saved.type||'AUTO')===type?'selected':''}>${type==='AUTO'?'Export default (recommended)':type}</option>`).join('')}</select></label>
    ${controls?`<div class="grid three">${controls}</div>`:''}
    <details><summary>Scheduler overrides and effective configuration</summary><label>Overrides · JSON<textarea name="image_scheduler_json" class="json" rows="5">${esc(JSON.stringify(overrides,null,2))}</textarea></label><pre>${esc(JSON.stringify(info.effective_config||{},null,2))}</pre></details>
    <div class="actions"><button type="button" id="reset-image-scheduler" class="secondary">Use export scheduler defaults</button>${info.qwen_reference?'<button type="button" id="qwen-reference-settings" class="secondary">Use Qwen official defaults</button>':''}</div>
    `;
}
function bindImageScheduler(form,m){
  const field=form.elements.image_scheduler_json,type=form.elements.image_scheduler_type,controls=[...form.querySelectorAll('[data-diffusion-key]')];
  const read=()=>{const v=JSON.parse(field.value);if(!v||Array.isArray(v)||typeof v!=='object')throw new Error('Scheduler overrides must be a JSON object');return v;};
  const sync=()=>{try{const values=read();for(const c of controls)c.value=values[c.dataset.diffusionKey]==null?'':String(values[c.dataset.diffusionKey]);field.setCustomValidity('');}catch(e){field.setCustomValidity(e.message);}};
  field.oninput=sync;
  for(const input of controls)input.onchange=input.oninput=()=>{try{const v=read(),k=input.dataset.diffusionKey;if(input.value==='')delete v[k];else v[k]=input.dataset.kind==='boolean'?input.value==='true':Number(input.value);field.value=JSON.stringify(v,null,2);field.setCustomValidity('');}catch(e){field.setCustomValidity(e.message);}};
  $('#reset-image-scheduler').onclick=()=>{type.value='AUTO';field.value='{}';sync();};
  if(m.diffusion_scheduler_info?.qwen_reference)$('#qwen-reference-settings').onclick=()=>{
    type.value='AUTO';field.value='{}';sync();
    for(const [k,v]of Object.entries({width:2048,height:2048,steps:40,seed:-1}))form.elements['gen_'+k].value=String(v);
    for(const axis of ['width','height'])form.elements['max_'+axis].value='2752';
    form.elements.image_negative_prompt_enabled.checked=false;form.elements.image_negative_prompt_enabled.onchange();
    form.elements.gen_max_sequence_length.value='';form.elements.gen_guidance_scale.value='';
    $('#image-size-preset').value='';
  };
}
function syncModelEditorTriggers(){
  for(const button of $$('[data-edit]')){
    button.setAttribute('aria-controls','model-editor');
    button.setAttribute('aria-haspopup','dialog');
    button.setAttribute('aria-expanded',String(button.dataset.edit===selectedModel&&$('#model-editor').open));
  }
}
function editModel(id,initialTab='general'){
  selectedModel=id;
  const m=state.models.find(m=>m.id===id),g=m.generation||{},isText=m.task==='text-generation',sizes=m.image_size_constraints,presets=sizes?.presets||[],qwenSampling=/qwen[ ._-]?3[ ._-]?8/i.test([m.id,m.name,m.path,m.inspection?.architecture].join(' '));
  const field=(name,label,value,type='number')=>`<label>${label}<input name="${name}" type="${type}" value="${esc(value)}" ${type==='number'?'step="any"':''}></label>`;
  if(!m)return;
  $('#model-editor').innerHTML=`<form id="model-form" class="settings-form">
    <div class="panel-heading modal-heading"><span class="modal-symbol">${UI.icon('tune')}</span><div><h2 id="model-settings-title">${esc(m.id)} · Settings</h2><p class="fine">${esc(m.name||m.id)} · OpenVINO</p></div>${UI.button('x','Close model settings','id="close-editor"')}</div>
    <div class="settings-layout">${UI.tabNavigation()}<div class="settings-scroll">${UI.section('general')}<h3>Make it yours</h3>
    <div class="model-availability"><label class="checkbox"><input type="checkbox" name="enabled" ${m.enabled!==false?'checked':''}> Expose this model through the API</label>${!isText?`<label class="checkbox"><input type="checkbox" name="default_image_tool" ${m.default_image_tool?'checked':''}> Use as default image tool</label>`:''}</div>
    <div class="grid three">${field('name','Display name',m.name,'text')}
      ${field('idle_timeout_seconds','Idle unload (seconds)',m.idle_timeout_seconds||0)}
      <label>Pipeline<select name="pipeline"><option value="llm" ${m.pipeline==='llm'?'selected':''}>LLM</option><option value="vlm" ${['vlm','vlm-text'].includes(m.pipeline)?'selected':''}>VLM · text + images</option><option value="image" ${m.pipeline==='image'?'selected':''}>${m.capabilities?.image_editing?'Image generation + editing':'Image generation'}</option></select></label>
      ${isText?field('context_limit','Context limit (tokens)',m.context_limit):field('max_width','Maximum width (px)',m.max_width||1024)}
    </div>
    <label>Exported model directory<div class="input-action"><input name="path" id="edit-path" value="${esc(m.path)}"><button type="button" data-browse="edit-path">Browse</button></div></label>
    ${!isText?`<h2>Memory after image requests</h2><label>Image model retention<select name="keep_image_model_loaded"><option value="auto" ${(m.keep_image_model_loaded??'auto')==='auto'?'selected':''}>Automatic (default)</option><option value="false" ${m.keep_image_model_loaded===false?'selected':''}>Unload after each request</option><option value="true" ${m.keep_image_model_loaded===true?'selected':''}>Keep loaded between requests</option></select></label><label>Text model after images<select name="restore_text_after_image"><option value="false" ${!m.restore_text_after_image?'selected':''}>Load on next request (default)</option><option value="true" ${m.restore_text_after_image?'selected':''}>Restore previous text models</option></select></label>`:''}
    </section>${UI.section('devices')}<h3>Where your model runs</h3><p class="fine">Choose Automatic, a single device, or an explicit split. Split workers reserve each participating device's planning budget; actual usage can differ.</p>${placementEditor(m)}
    <label>Memory reservation (GiB)<div class="input-action"><input name="memory_gib" type="number" min="0.1" max="9999" step="any" value="${esc(m.memory_gib)}" required><button type="button" class="secondary" data-memory-default>Default</button></div></label>
    </section>${UI.section('generation')}<h2>Default generation settings</h2>
    ${isText&&qwenSampling?`<label>Qwen3.8 sampling preset<select id="sampling-preset"><option value="">Keep current settings</option>${(state.sampling_presets||[]).map(p=>`<option value="${esc(p.id)}">${esc(p.label)}</option>`).join('')}</select></label>`:''}
    <div class="grid three">${isText?field('gen_max_tokens','Output token budget',g.max_tokens??8192)+field('gen_temperature','Temperature',g.temperature??'')+field('gen_top_p','Top P',g.top_p??'')+field('gen_top_k','Top K',g.top_k??'')+field('gen_min_p','Min P',g.min_p??'')+field('gen_presence_penalty','Presence penalty',g.presence_penalty??'')+field('gen_frequency_penalty','Frequency penalty',g.frequency_penalty??'')+field('gen_repetition_penalty','Repetition penalty',g.repetition_penalty??''):field('gen_width','Default width (px)',g.width??512)+field('gen_height','Default height (px)',g.height??512)+field('gen_steps','Inference steps',g.steps??20)+field('gen_guidance_scale','Guidance scale',g.guidance_scale??'')+field('gen_seed','Seed',g.seed??-1)+field('max_height','Maximum height (px)',m.max_height||1024)}</div>
    
    ${!isText?`
    ${presets.length?`<div class="grid two"><label>Qwen-Image-2.1 size preset<select id="image-size-preset"><option value="">Custom / keep current dimensions</option>${presets.map(p=>`<option value="${p.width}x${p.height}">${esc(p.aspect_ratio)} · ${p.width} × ${p.height}</option>`).join('')}</select></label><div class="field-actions"><button type="button" id="allow-image-presets" class="secondary">Allow all listed preset sizes</button></div></div>`:''}
    <label class="checkbox"><input type="checkbox" name="image_negative_prompt_enabled" ${Object.hasOwn(g,'negative_prompt')?'checked':''}> Enable default negative prompt</label>
    <label>Negative prompt<textarea name="gen_negative_prompt" rows="3" placeholder="Details to avoid">${esc(g.negative_prompt??'')}</textarea></label>
    
    ${presets.length?`<label>Text encoder token limit<input name="gen_max_sequence_length" type="number" min="1" max="1048576" step="1" value="${esc(g.max_sequence_length??'')}" placeholder="Native default · 8192"></label>`:''}`:''}
    ${!isText&&m.image_editing?.supports_strength?`<h2>Image editing</h2>${field('gen_strength','Default edit strength',g.strength??.8)}`:''}
    ${isText?`<h2>Multi-token prediction (MTP)</h2>
      <div class="grid three"><label>MTP decoding<select name="mtp_enabled"><option value="false" ${!m.mtp?.enabled?'selected':''}>Disabled</option><option value="true" ${m.mtp?.enabled?'selected':''} ${!m.capabilities?.mtp_available&&!m.mtp?.enabled?'disabled':''}>Enabled · bundled MTP head</option></select></label>
      <label>MTP device<select name="mtp_device">${mtpDeviceOptions(m.mtp?.device)}</select></label>
      <label>Draft tokens per step<input name="mtp_num_assistant_tokens" type="number" min="1" max="64" step="1" value="${esc(m.mtp?.num_assistant_tokens??2)}"></label></div>
      `:''}
    </section>${UI.section('runtime')}${isText?`<h2>KV cache and context memory</h2>
      
      <div class="grid three">${runtimeSection('kv',m)}</div>
      
      <details><summary>Advanced KV layout</summary><div class="grid three">${runtimeSection('kv_advanced',m)}</div></details>`:''}
    ${!isText?imageSchedulerSection(m):''}
    <h2>OpenVINO runtime</h2><div class="grid three">${runtimeSection('runtime',m)}</div>
    <details class="cpu-runtime"><summary>CPU scheduling</summary><p class="fine">Applies when the main model uses CPU, including CPU offload. Blank values keep runtime defaults; device and OS support still applies.</p><div class="grid three">${runtimeSection('cpu_runtime',m)}</div></details>
    
    </section>${UI.section('advanced')}<details open><summary>Advanced properties and chat template · JSON</summary>
      
      <div class="grid two"><label>OpenVINO plugin properties · JSON<textarea class="json" name="plugin_config">${esc(JSON.stringify(m.plugin_config||{},null,2))}</textarea></label>
        ${isText?`<label>Scheduler / KV cache · JSON<textarea class="json" name="scheduler_config">${esc(JSON.stringify(m.scheduler_config||{},null,2))}</textarea></label>`:''}</div>
      ${isText?`<label>Chat template settings · JSON<textarea class="json" rows="3" name="template_context">${esc(JSON.stringify(m.template_context||{},null,2))}</textarea></label>`:''}
    </details>
    <button class="text-button danger" type="button" id="remove-model">Remove from registry</button></section></div></div>
    <div class="settings-footer"><button type="button" class="secondary" data-modal-close>Cancel</button><button class="primary" type="submit">${UI.icon('check')} Save model settings</button></div></form>`;
  syncModelEditorTriggers();
  $('#close-editor').onclick=()=>UI.close($('#model-editor'));
  UI.tabs($('#model-form'),initialTab);
  bindBrowse();bindRuntimeControls($('#model-form'));bindMemoryDefault($('#model-form'));
  const readPlacement=bindPlacementEditor($('#model-form'),m);
  if(isText){
    const f=$('#model-form');
    for(const key of ['temperature','top_p','top_k','min_p','presence_penalty','frequency_penalty','repetition_penalty']){
      f.elements['gen_'+key].placeholder=key==='temperature'?(m.mtp?.enabled?'Default · 0 (MTP)':'Default · 0.7'):'Model / runtime default';
    }
    if($('#sampling-preset'))$('#sampling-preset').onchange=e=>{
      const preset=(state.sampling_presets||[]).find(p=>p.id===e.target.value);if(!preset)return;
      try{
        const context=JSON.parse(f.elements.template_context.value);
        if(!context||typeof context!=='object'||Array.isArray(context))throw new Error('Chat template settings must be a JSON object');
        const merged={...context,...preset.template_context};
        if(merged.reasoning_effort==='none')delete merged.reasoning_effort;
        f.elements.template_context.value=JSON.stringify(merged,null,2);
        for(const [key,value] of Object.entries(preset.generation))f.elements['gen_'+key].value=String(value);
        f.elements.gen_frequency_penalty.value='0';
        toast(f.elements.mtp_enabled.value==='true'?'Preset filled. Disable MTP to use sampled decoding, or keep MTP and set temperature to 0.':'Preset filled. Save model settings to apply.');
      }catch(error){toast(error.message);}
    };
  }
  if(isText)$('#model-form').elements.mtp_enabled.onchange=()=>{
    const f=$('#model-form');if(f.elements.mtp_enabled.value==='true'){
      f.elements.gen_temperature.value=0;
      f.elements.gen_temperature.placeholder='Default · 0 (MTP)';
      const prefix=f.elements.runtime_enable_prefix_caching;prefix.value='false';prefix.dispatchEvent(new Event('change'));
    }else f.elements.gen_temperature.placeholder='Default · 0.7';
  };
  if(!isText){
    const f=$('#model-form');
    bindImageScheduler(f,m);
    for(const axis of ['width','height']){
      f.elements['gen_'+axis].min=String(sizes?.minimum||256);
      f.elements['gen_'+axis].step=String(sizes?.multiple_of||64);
      f.elements['max_'+axis].min=String(sizes?.minimum||256);
      f.elements['max_'+axis].step='1';
    }
    const negative=f.elements.image_negative_prompt_enabled;
    negative.onchange=()=>{f.elements.gen_negative_prompt.disabled=!negative.checked;};negative.onchange();
    if(presets.length){
      $('#image-size-preset').onchange=e=>{
        const preset=presets.find(p=>`${p.width}x${p.height}`===e.target.value);if(!preset)return;
        for(const axis of ['width','height']){
          f.elements['gen_'+axis].value=String(preset[axis]);
          f.elements['max_'+axis].value=String(Math.max(Number(f.elements['max_'+axis].value),preset[axis]));
        }
      };
      $('#allow-image-presets').onclick=()=>{
        for(const axis of ['width','height'])f.elements['max_'+axis].value=String(Math.max(Number(f.elements['max_'+axis].value),...presets.map(p=>p[axis])));
      };
    }
  }
  $('#model-form').onsubmit=async e=>{
    e.preventDefault();
    try{
      const raw=formObject(e.target),body={model:id,generation:{...g},enabled:e.target.elements.enabled.checked};
      for(const[k,v]of Object.entries(raw)){
        if(k.startsWith('runtime_')||k.startsWith('mtp_')||k.startsWith('image_')||k.startsWith('placement_'))continue;
        if(k==='gen_negative_prompt')body.generation.negative_prompt=v;
        else if(k==='gen_max_sequence_length'&&v.trim()==='')delete body.generation.max_sequence_length;
        else if(k.startsWith('gen_')){if(v.trim()==='')delete body.generation[k.slice(4)];else body.generation[k.slice(4)]=Number(v);}
        else if(['plugin_config','scheduler_config','template_context'].includes(k))body[k]=UI.parseJSON(e.target,k);
        else if(k!=='enabled'&&k!=='default_image_tool')body[k]=v;
      }
      const placement=readPlacement();if(placement){body.placement=placement;body.device=placement.devices[0];}
      if(!isText)body.default_image_tool=e.target.elements.default_image_tool.checked;
      if(isText)body.mtp={enabled:raw.mtp_enabled==='true',num_assistant_tokens:Number(raw.mtp_num_assistant_tokens),...(raw.mtp_device?{device:raw.mtp_device}:{})};
      else {body.diffusion_scheduler={type:raw.image_scheduler_type,config:UI.parseJSON(e.target,'image_scheduler_json')};body.keep_image_model_loaded=raw.keep_image_model_loaded==='auto'?'auto':raw.keep_image_model_loaded==='true';body.restore_text_after_image=raw.restore_text_after_image==='true';if(!e.target.elements.image_negative_prompt_enabled.checked)delete body.generation.negative_prompt;}
      numbers(body,['memory_gib','idle_timeout_seconds','context_limit','max_width','max_height']);
      if(body.path===m.path)delete body.path;
      await api('/api/settings',body);UI.close($('#model-editor'));
      await refresh(true);toast('Settings saved. Click Load or send a request to apply them.');
      $$('[data-edit]').find(button=>button.dataset.edit===id)?.focus({preventScroll:true});
    }catch(e){toast(e.message);}
  };
  $('#remove-model').onclick=async()=>{
    if(!confirm(`Remove ${id} from the registry? Model files will remain on disk.`))return;
    try{await api(`/api/models/${id}/remove`,{});selectedModel=null;UI.close($('#model-editor'));await refresh(true);}catch(e){toast(e.message);}
  };
  window.Help.install();UI.open($('#model-editor'));syncModelEditorTriggers();
}
function renderDevices(){
  $('#devices-grid').innerHTML=state.scheduler.devices.map(d=>{
    const budget=Number(d.budget_gib)||0,reserved=Number(d.reserved_gib)||0,percent=budget?Math.max(0,Math.min(100,reserved/budget*100)):0;
    return `<article class="panel device-overview"><div class="device-title"><span class="model-avatar">${UI.icon(d.device_type==='cpu'||d.id==='CPU'?'chip':'gpu')}</span><span class="device-id">${esc(d.id)}</span>${UI.button('gear','Settings for '+d.id,`data-edit-device="${esc(d.id)}"`)}</div><h2>${esc(d.name)}</h2><p class="fine">${d.device_type==='cpu'||d.id==='CPU'?(d.detected===false?'CPU · Not detected':'CPU · Runtime available'):d.support_level==='experimental'?'Experimental':d.device_type==='discrete'?'Dedicated graphics':'Integrated / shared graphics'}</p><div class="reservation-ring" data-reservation="${percent}"><div><strong>${fmtMemory(reserved)}</strong><small>GiB RESERVED</small></div></div><div class="device-numbers"><span>${d.device_type==='discrete'?'Dedicated VRAM':d.device_type==='cpu'?'System RAM':d.device_type==='integrated'?'Reported shared capacity':'Reported capacity'}</span><strong>${d.total_memory_gib!=null?fmtMemory(d.total_memory_gib)+' GiB':'Not reported'}</strong></div><div class="device-numbers"><span>Planning budget</span><strong>${fmtMemory(budget)} GiB</strong></div><div class="device-overview-footer">${status(state.scheduler.busy.includes(d.id)?'busy':'idle')}<span>Scheduling estimate · usage unavailable</span></div></article>`;
  }).join('')||'<div class="panel">No devices detected. Install OpenVINO and restart the server.</div>';
  $$('[data-reservation]').forEach(el=>el.style.setProperty('--reservation',el.dataset.reservation+'%'));
  $$('[data-edit-device]').forEach(button=>button.onclick=()=>editDevice(button.dataset.editDevice));
}
function editDevice(id){
  const d=state.scheduler.devices.find(device=>device.id===id);if(!d)return;
  $('#device-settings-dialog').innerHTML=`<form class="settings-form device-form" data-device="${esc(d.id)}">
    <div class="modal-heading"><div><h2 id="device-settings-title">Device settings</h2><p class="fine">${esc(d.id)} · ${esc(d.name)}</p></div>${UI.button('x','Close device settings','data-modal-close')}</div><div class="settings-scroll"><div class="device-title"><div><div class="eyebrow">${esc(d.id)}</div><h2>${esc(d.name)}</h2></div>${status(state.scheduler.busy.includes(d.id)?'busy':'idle')}</div>
    ${d.support_level?`<p class="fine device-support"><strong>${d.support_level==='supported'?(d.id.split('.')[0]==='CPU'?'CPU · Runtime available':'Supported target · Intel'):'Experimental'}</strong> · ${esc(d.support_note)}</p>`:''}
    <div class="device-numbers"><span>Scheduling reservation</span><span>${fmtMemory(d.reserved_gib)} / ${fmtMemory(d.budget_gib)} GiB</span></div>
    <div class="memory-track"><div data-width="${Math.min(100,100*d.reserved_gib/d.budget_gib)}"></div></div>
    <div class="grid two"><label>Planning budget (GiB)<input name="budget_gib" type="number" min="0.1" step="0.1" value="${d.budget_gib}"></label>
      <label>Model residency<select name="policy"><option value="exclusive" ${d.policy==='exclusive'?'selected':''}>Unload on switch</option><option value="budget" ${d.policy==='budget'?'selected':''}>Keep within memory budget (default)</option></select></label>
      ${d.device_type==='cpu'||d.id.split('.')[0]==='CPU'?`<div class="device-threads"><label>CPU inference threads<div class="input-action"><input name="inference_threads" type="number" min="1" max="2147483647" step="1" value="${esc(d.inference_threads??'')}" placeholder="Auto" aria-describedby="cpu-threads-note-${esc(d.id)}"><button type="button" class="secondary" data-cpu-threads-auto aria-label="Use automatic CPU thread count">Auto</button></div></label><p class="fine" id="cpu-threads-note-${esc(d.id)}">Auto lets the runtime choose. Applies to CPU inference and CPU offload unless a model overrides it.</p></div>`:''}</div>
    <p class="fine">This budget plans model residency and screens full-GPU loads. It is not a hard memory limit and is not passed to the splitter as a partition size. A budget above dedicated VRAM permits a native shared-memory attempt when the driver supports it.</p><p class="fine">Reserved GiB is a scheduling estimate. Measured GPU usage is unavailable; check your system monitor for actual dedicated and shared memory use.</p>
    </div><div class="settings-footer"><button type="button" class="secondary" data-modal-close>Cancel</button><button class="primary">${UI.icon('check')} Save device settings</button></div></form>`;
  $$('[data-width]').forEach(el=>el.style.width=el.dataset.width+'%');
  $$('.device-form').forEach(f=>{
    const threads=f.elements.inference_threads;
    if(threads)f.querySelector('[data-cpu-threads-auto]').onclick=()=>{threads.value='';threads.focus();};
    f.onsubmit=async e=>{
      e.preventDefault();
      try{
        const body={device:f.dataset.device,...numbers(formObject(f),['budget_gib'])};
        if(threads)body.inference_threads=threads.value.trim()===''?null:Number(threads.value);
        await api('/api/settings',body);UI.close($('#device-settings-dialog'));await refresh(true);toast('Device settings saved');
      }catch(e){toast(e.message);}
    };
  });
  window.Help.install();UI.open($('#device-settings-dialog'));
}
function renderServer(){const endpoints=[['GET','/v1/models'],['POST','/v1/chat/completions'],['POST','/v1/images/generations'],['POST','/v1/images/edits'],['GET','/api/artifacts'],['GET','/api/image/jobs'],['POST','/api/context']];$('#endpoints').innerHTML=endpoints.map(([method,path])=>`<div class="endpoint"><strong>${method}</strong><code>${esc(location.origin+path)}</code>${UI.button('copy','Copy URL',`data-copy="${path}"`)}</div>`).join('');$$('[data-copy]').forEach(b=>b.onclick=()=>copy(location.origin+b.dataset.copy));const f=$('#server-form'),s=state.server;for(const k of ['host','port','models_dir','load_timeout_seconds','request_timeout_seconds','download_max_gib','tls_cert','tls_key'])f.elements[k].value=s[k]??({host:'127.0.0.1',port:8765,load_timeout_seconds:900,request_timeout_seconds:3600,download_max_gib:100}[k]||'');f.elements.browser_access.value=s.browser_access||'origins';f.elements.allowed_origins.value=(s.allowed_origins||[]).join('\n');f.elements.terminal_metrics.checked=s.terminal_metrics!==false;renderConnections();$('#api-key-state').textContent=s.api_key_set?'· configured':'· not configured';$('#restart-note').textContent=s.restart_required?'Restart required for address, port or HTTPS changes':'';exampleModels();bindBrowse();}
function exampleModels(){$('#api-model').innerHTML=modelOptions($('#example-type').value==='chat'?'text-generation':'text-to-image');renderExample();}
function renderExample(){const editing=$('#example-type').value==='edit',image=editing||$('#example-type').value==='image',path=editing?'/v1/images/edits':image?'/v1/images/generations':'/v1/chat/completions';const body=image?{model:$('#api-model').value||'image-model',prompt:editing?'Add a red scarf and preserve the subject.':'A calm futuristic workspace',size:'auto',response_format:'b64_json',...(editing?{image:'data:image/png;base64,BASE64_ENCODED_PNG'}:{})}:{model:$('#api-model').value||'llm-model',messages:[{role:'user',content:'Hello'}],max_tokens:256,stream:false};$('#api-example').textContent=`POST ${location.origin+path}\nContent-Type: application/json\n${state.server.api_key_set?'Authorization: Bearer YOUR_API_KEY\n':''}\n${JSON.stringify(body,null,2)}`;}

async function copy(text){try{await navigator.clipboard.writeText(text);toast('Copied');return true;}catch{toast('Clipboard unavailable. Select and copy the text.');return false;}}
function renderDownloads(){
 const openLogs=new Set($$('[data-download-log][open]').map(el=>el.dataset.downloadLog));
 const device=$('#download-device'),old=device.value;
 const signature=JSON.stringify(state.scheduler.devices.map(d=>[d.id,d.name,d.support_level,d.detected]));if(device.dataset.devices!==signature){device.innerHTML=deviceOptions(old);device.dataset.devices=signature;}const form=$('#download-form');bindMemoryDefault(form,!form.dataset.memoryReady||device.value!==old);form.dataset.memoryReady='true';device.required=form.elements.memory_gib.required=$('#download-register').checked;
 const phases={'queued':'Queued','downloading':'Downloading files','preparing-runtime':'Installing compatible runtime','waiting-converter':'Waiting for converter','converting':'Converting to OpenVINO','validating':'Validating export','completed':'Saved','failed':'Failed','cancelled':'Cancelled','interrupted':'Interrupted'};
 $('#download-jobs').innerHTML=state.downloads.map(j=>`<div class="panel download-row"><div class="panel-heading"><h2>${esc(j.destination.split(/[\\/]/).pop())}</h2>${status(j.status)}</div><code>${esc(j.destination)}</code><p>${esc(phases[j.status]||j.status)}${j.profile?' · '+esc(j.profile):''}${j.converted?' · converted to '+esc(j.precision.toUpperCase()):''}</p>${!done(j.status)?`<progress class="download-progress" ${j.status==='downloading'&&j.total_bytes?'value="'+j.bytes+'" max="'+j.total_bytes+'"':''}></progress><button class="text-button" data-cancel-download="${j.id}">Cancel</button>`:''}${j.retry&&!done(j.status)?``:''}${j.error?`<p class="fine danger">${esc(j.error)}</p>`:''}${j.registration_error?`<p class="fine danger">${esc(j.registration_error)}</p>`:''}${j.log?.length?`<details data-download-log="${j.id}" ${openLogs.has(j.id)?'open':''}><summary>Conversion log</summary><pre class="conversion-log">${esc(j.log.join('\n'))}</pre></details>`:''}${j.model_id?`<p>Registered as <strong>${esc(j.model_id)}</strong>. Open Models and click Load.</p>`:j.status==='completed'?`<button class="secondary" data-register-download="${j.id}">Register saved model</button>`:''}</div>`).join('');
 $$('[data-cancel-download]').forEach(b=>b.onclick=async()=>{try{await api('/api/downloads/'+b.dataset.cancelDownload+'/cancel',{});await refresh();}catch(e){toast(e.message);}});
 $$('[data-register-download]').forEach(b=>b.onclick=()=>openRegister(state.downloads.find(j=>j.id===b.dataset.registerDownload).destination));
}
function renderActivity(){const opened=new Set($$('[data-tool-log][open]').map(el=>el.dataset.toolLog)),loads=new Set($$('[data-load-log][open]').map(el=>el.dataset.loadLog));$('#activity-list').innerHTML=state.jobs.slice(0,10).map(j=>`<div class="activity-row"><time>${new Date(j.created*1000).toLocaleTimeString([],{hour12:false})}</time><div><strong>${esc(j.model)}</strong> · ${esc(j.phase||j.status)}<p class="${j.error?'error':''}">${esc(j.error||j.prompt.slice(0,160))}</p>${j.context?`<p>Input ${fmt(j.context.input_tokens)} · output ${fmt(j.context.generated_tokens)} · context ${fmt(j.context.used_tokens)}/${fmt(j.context.context_limit)} · ${esc(j.context.accuracy)}</p>`:''}${loadAttemptsHTML(j,loads.has(j.id))}${j.tool_attempts?.length?`<details data-tool-log="${esc(j.id)}" ${opened.has(j.id)?'open':''}><summary>LLM image handoff · ${j.tool_attempts.length} attempt(s)</summary><pre>${esc(JSON.stringify(j.tool_attempts,null,2))}</pre></details>`:''}</div><div>${status(j.status)}${j.request?.max_tokens||j.request?.width?`<button class="text-button" data-view-job="${esc(j.id)}">${done(j.status)?'View result':'View progress'}</button>`:''}${!done(j.status)?`<button class="text-button" data-cancel-job="${j.id}">Cancel</button>`:''}</div></div>`).join('')||'';$$('[data-copy-load]').forEach(b=>b.onclick=async()=>{try{const job=await api('/api/jobs/'+b.dataset.copyLoad);await copy(JSON.stringify({model:job.model,plan:job.placement_plan,attempts:job.placement_attempts,error:job.error},null,2));}catch(error){toast(error.message);}});$$('[data-view-job]').forEach(b=>b.onclick=()=>{show('chat');window.Chat.recover(b.dataset.viewJob);});$$('[data-cancel-job]').forEach(b=>b.onclick=async()=>{try{await api('/api/jobs/'+b.dataset.cancelJob+'/cancel',{});await refresh();}catch(e){toast(e.message);}});}
function loadAttemptsHTML(job,open){
  if(!job.placement_plan)return '';
  const names={not_attempted:'Not attempted',skipped:'Skipped before loading',loading:'Loading',failed:'Failed',succeeded:'Loaded',cancelled:'Cancelled'},plan=job.placement_plan;
  const reasons={weight_budget:'Stored weights exceed planning budget',reservation_budget:'Reservation exceeds planning budget',gpu_kernel_build_failed:'GPU compilation failure',split_inference_failed:'Inference probe failure',host_memory_pressure:'System RAM guard',cleanup_failed:'Worker cleanup failure',native_load_rejected:'Native load failure',load_timeout:'Load deadline exceeded'};
  return `<details class="load-attempts" data-load-log="${esc(job.id)}" ${open?'open':''}><summary>Load plan & attempts</summary><p>${esc({auto:'Automatic',single:'Single device',split:'Split across selected devices',manual:'Legacy manual split'}[plan.mode]||plan.mode)}${plan.preferred_reused?' · Reusing the last successful placement':''}</p><p>${esc(plan.split?.reason||'')}</p>${plan.unavailable_devices?.length?`<p>Not attempted · not detected: ${esc(plan.unavailable_devices.join(', '))}</p>`:''}<ol>${(job.placement_attempts||[]).map(a=>`<li><div><strong>${esc(a.target)}</strong><span>${esc(names[a.status]||a.status)}</span></div><p>${esc(a.label)}${a.reason_code?' · '+esc(reasons[a.reason_code]||a.reason_code):''}</p>${a.reason?`<pre>${esc(a.reason)}</pre>`:''}</li>`).join('')}</ol><button type="button" class="secondary load-copy" data-copy-load="${esc(job.id)}">${UI.icon('copy')} Copy load details</button></details>`;
}
function openRegister(path=''){$('#register-form').reset();$('#register-device').innerHTML=deviceOptions();bindMemoryDefault($('#register-form'),true);$('#register-path').value=path;$('#inspection-result').textContent='';if(path){const name=path.split(/[\\/]/).filter(Boolean).pop();$('#register-id').value=name.replace(/[^A-Za-z0-9_.-]/g,'-').slice(0,80);$('#register-name').value=name;}$('#register-dialog').showModal();}
function bindBrowse(){$$('[data-browse]').forEach(b=>b.onclick=async()=>{
  folderTarget=b.dataset.browse;
  $('#folder-title').textContent=({'register-path':'Choose a model folder','edit-path':'Choose a model folder','download-destination':'Choose an output folder','models-dir':'Choose the default models folder'})[folderTarget]||'Choose a folder';
  try{await listFolders($('#'+folderTarget).value||state.server.models_dir||undefined);$('#folder-dialog').showModal();}
  catch{try{await listFolders();$('#folder-dialog').showModal();}catch(e){toast(e.message);}}
});}
async function listFolders(path){folderState=await api('/api/folders'+(path?'?path='+encodeURIComponent(path):''));$('#folder-path').value=folderState.path;$('#folder-list').innerHTML=folderState.folders.map(f=>`<button data-folder="${esc(f.path)}">▱ ${esc(f.name)}</button>`).join('')||'';$('#folder-roots').innerHTML=folderState.roots.map(p=>`<button class="secondary" data-folder="${esc(p)}">${esc(p)}</button>`).join('');$$('[data-folder]').forEach(b=>b.onclick=()=>listFolders(b.dataset.folder).catch(e=>toast(e.message)));}
function renderConnections(){const errors=state.server.connection_errors||[];$('#connection-errors').textContent=errors.length?errors.slice(-5).reverse().map(e=>`${new Date(e.time*1000).toLocaleTimeString()} · ${e.origin} · ${e.path}\n${e.reason}`).join('\n\n'):'No rejected browser origins since startup.';}

$('.sidebar .brand').onclick=e=>{if(e.button!==0||e.ctrlKey||e.metaKey||e.shiftKey||e.altKey||!state)return;e.preventDefault();show('models');};
$$('[data-view]').forEach(b=>b.onclick=()=>show(b.dataset.view));$('#add-model').onclick=()=>openRegister();$('#open-download').onclick=()=>{show('downloads');if(!$('#download-destination').value)$('#download-destination').value=(state.server.models_dir||'')+'/new-model';};$$('.close-dialog').forEach(b=>b.onclick=()=>b.closest('dialog').close());$('#folder-go').onclick=()=>listFolders($('#folder-path').value).catch(e=>toast(e.message));$('#folder-up').onclick=()=>listFolders(folderState.parent).catch(e=>toast(e.message));$('#folder-select').onclick=()=>{$('#'+folderTarget).value=folderState.path;$('#folder-dialog').close();};
$('#inspect-folder').onclick=async()=>{
  const field=$('#register-path');
  $('#inspection-result').textContent='';
  if($('#register-dialog').contains($('#toast')))hideToast();
  if(!field.value.trim()){field.focus();toast('Choose a model folder first.');return;}
  try{const r=await api('/api/models/inspect',{path:field.value});$('#inspection-result').textContent=`${r.architecture} · ${r.ir_components} IR components · ${fmtMemory(r.size_bytes/1024**3)} GiB on disk. ${r.validation}`;$('#register-pipeline').value=r.pipeline;$('#register-form').elements.context_limit.value=Math.min(32768,r.declared_context_limit||32768);if(!$('#register-id').value)$('#register-id').value=r.path.split(/[\\/]/).pop().replace(/[^A-Za-z0-9_.-]/g,'-').slice(0,80);}catch(e){toast(e.message);}
};
$('#register-form').onsubmit=async e=>{e.preventDefault();try{const body=numbers(formObject(e.target),['memory_gib','context_limit']);if(!body.pipeline)delete body.pipeline;await api('/api/models',body);$('#register-dialog').close();show('models');await refresh(true);toast('Model registered. Use Load to validate it with OpenVINO.');}catch(e){toast(e.message);}};
$('#download-form').onsubmit=async e=>{e.preventDefault();try{const body=numbers(formObject(e.target),['memory_gib']);body.register=$('#download-register').checked;await api('/api/downloads',body);toast('Model import started');await refresh();}catch(e){toast(e.message);}};
function hideApiKey(){$('#server-api-key').type='password';$('#show-api-key').textContent='Show';$('#show-api-key').setAttribute('aria-pressed','false');}
$('#generate-api-key').onclick=async()=>{const button=$('#generate-api-key');button.disabled=true;try{const result=await api('/api/server/api-key',{});$('#server-api-key').value=result.api_key;hideApiKey();toast('New key generated. Copy it, then Save server settings to activate it.');}catch(error){toast(error.message);}finally{button.disabled=false;}};
$('#copy-api-key').onclick=async()=>{const field=$('#server-api-key'),value=field.value||key;if(!value){toast('Generate or enter an API key first.');return;}try{await navigator.clipboard.writeText(value);toast('API key copied');}catch{field.value=value;field.type='text';field.focus();field.select();$('#show-api-key').textContent='Hide';$('#show-api-key').setAttribute('aria-pressed','true');toast('Clipboard unavailable. The key is selected; press Ctrl+C to copy.');}};
$('#show-api-key').onclick=()=>{const field=$('#server-api-key'),show=field.type==='password';field.type=show?'text':'password';$('#show-api-key').textContent=show?'Hide':'Show';$('#show-api-key').setAttribute('aria-pressed',String(show));};
$('#server-form').onsubmit=async e=>{e.preventDefault();try{const body=numbers(formObject(e.target),['port','load_timeout_seconds','request_timeout_seconds','download_max_gib']);body.terminal_metrics=e.target.elements.terminal_metrics.checked;body.allowed_origins=body.allowed_origins.split('\n').map(s=>s.trim()).filter(Boolean);if(!body.api_key)delete body.api_key;await api('/api/server',body);if(body.api_key){key=body.api_key;sessionStorage.setItem('lazyrig.key',key);e.target.elements.api_key.value='';hideApiKey();}UI.close($('#server-settings-dialog'));await refresh(true);toast('Server settings saved');}catch(e){toast(e.message);}};
$('#example-type').onchange=exampleModels;$('#api-model').onchange=renderExample;$('#copy-example').onclick=()=>copy($('#api-example').textContent);
$('#set-key').onclick=()=>$('#key-dialog').showModal();$('#key-form').onsubmit=async e=>{e.preventDefault();key=$('#client-key').value.trim();sessionStorage.setItem('lazyrig.key',key);$('#key-dialog').close();await refresh(true);};bindBrowse();refresh(true);setInterval(()=>{if(!document.hidden)refresh();},2500);

$('#refresh-devices').onclick=async()=>{try{await api('/api/devices/refresh',{});await refresh(true);toast('Device detection refreshed');}catch(e){toast(e.message);}};
$('#download-register').onchange=()=>{$('#download-device').required=$('#download-form').elements.memory_gib.required=$('#download-register').checked;};
$('#download-url').onchange=()=>{try{const u=new URL($('#download-url').value),parts=u.pathname.split('/').filter(Boolean);if(u.hostname==='huggingface.co'&&parts.length>=2){const name=parts[1].replace(/[^A-Za-z0-9_.-]/g,'-');if(!$('#download-destination').value||$('#download-destination').value.endsWith('/new-model'))$('#download-destination').value=(state.server.models_dir||'models')+'/'+name;if(!$('#download-alias').value)$('#download-alias').value=name;}}catch{}};

$('#allow-last-origin').onclick=()=>{const origin=[...(state.server.connection_errors||[])].reverse().find(e=>/^https?:\/\//.test(e.origin))?.origin;if(!origin){toast('No HTTP(S) origin found. For a file or sandboxed frontend, use Any frontend with API key.');return;}const field=$('#server-form').elements.allowed_origins;const origins=new Set(field.value.split('\n').filter(Boolean));origins.add(new URL(origin).origin);field.value=[...origins].join('\n');toast('Origin added to the form. Save server settings to apply it.');};



function renderRuntimeUpdate(result){
  runtimeUpdateState=result;
  const names={'openvino':'OpenVINO™','openvino-genai':'OpenVINO GenAI','openvino-tokenizers':'OpenVINO Tokenizers'};
  const versions=Object.entries(result.installed||{}).map(([name,value])=>`<dt>${esc(names[name]||name)}</dt><dd><code>${esc(result.installed_builds?.[name]||value||'Not installed')}</code></dd>`);
  if(result.latest_release)versions.push(`<dt>Latest stable OpenVINO GenAI</dt><dd><code>${esc(result.latest_release)}</code></dd>`);
  if(result.target)versions.push(`<dt>Compatible OpenVINO GenAI</dt><dd><code>${esc(result.target)}</code></dd>`);
  $('#runtime-update-versions').innerHTML=versions.join('');
  const busy=result.status==='checking'||result.status==='updating'||!!runtimeUpdatePending;
  $('#runtime-update-install').disabled=!result.can_update||busy;
  $('#runtime-update-check').disabled=busy;
  $('#runtime-update-message').textContent=result.message||'Check for stable runtime updates.';
  const last=result.last_update;
  $('#runtime-update-last').textContent=last?`Last update: ${last.message||last.status}`:'';
}
function scheduleRuntimeUpdatePoll(){clearTimeout(runtimeUpdateTimer);runtimeUpdateTimer=setTimeout(pollRuntimeUpdate,2000);}
async function pollRuntimeUpdate(){
  if(runtimeUpdatePolling)return;
  if(!runtimeUpdatePending&&!$('#runtime-update-dialog').open)return;
  runtimeUpdatePolling=true;
  try{
    const result=await api('/api/runtime/update',undefined,10000), last=result.last_update;
    if(runtimeUpdatePending&&last?.id===runtimeUpdatePending&&last.status!=='updating'){
      runtimeUpdatePending='';sessionStorage.removeItem('lazyrig.runtime-update');
      toast(last.message||'Runtime update finished');await refresh();
    }
    if(runtimeUpdatePending&&result.status==='ready'&&result.check_id===runtimeUpdatePending&&last?.id!==runtimeUpdatePending){
      runtimeUpdatePending='';sessionStorage.removeItem('lazyrig.runtime-update');
      result.can_update=false;result.message='The update was not accepted by the server. Check again before retrying.';
    }
    if(runtimeUpdatePending&&result.status!=='updating')result.message='Runtime update is in progress. Waiting for the server to restart…';
    renderRuntimeUpdate(result);
    if(result.status==='checking'||runtimeUpdatePending)scheduleRuntimeUpdatePoll();
  }catch(error){
    $('#runtime-update-message').textContent=runtimeUpdatePending
      ?'Waiting for LazyRig to restart. Check the updater console or logs/runtime-update.log for progress. If it reports failure, repair the environment and start LazyRig again.'
      :error.message;
    if(runtimeUpdatePending)scheduleRuntimeUpdatePoll();
  }finally{runtimeUpdatePolling=false;}
}
async function checkRuntimeUpdate(){
  $('#runtime-update-install').disabled=true;$('#runtime-update-check').disabled=true;
  $('#runtime-update-message').textContent='Checking stable releases on PyPI…';
  try{renderRuntimeUpdate(await api('/api/runtime/check',{},10000));scheduleRuntimeUpdatePoll();}
  catch(error){$('#runtime-update-message').textContent=error.message;$('#runtime-update-check').disabled=false;}
}
$('#runtime-update-open').onclick=()=>{
  $('#runtime-update-dialog').showModal();
  if(runtimeUpdatePending)pollRuntimeUpdate();else checkRuntimeUpdate();
};
$('#runtime-update-check').onclick=checkRuntimeUpdate;
$('#runtime-update-install').onclick=async()=>{
  if(!runtimeUpdateState?.can_update)return;
  runtimeUpdatePending=runtimeUpdateState.check_id;
  sessionStorage.setItem('lazyrig.runtime-update',runtimeUpdatePending);
  $('#runtime-update-install').disabled=true;$('#runtime-update-check').disabled=true;
  try{renderRuntimeUpdate(await api('/api/runtime/update',{check_id:runtimeUpdatePending},10000));}
  catch(error){
    if(error.status){runtimeUpdatePending='';sessionStorage.removeItem('lazyrig.runtime-update');$('#runtime-update-check').disabled=false;}
    $('#runtime-update-message').textContent=error.status?error.message:'Waiting for the update receipt or server restart. Check the updater console for progress.';
  }
  scheduleRuntimeUpdatePoll();
};
if(runtimeUpdatePending)pollRuntimeUpdate();

$('#server-settings-open').onclick=()=>{renderServer();window.Help.install();UI.open($('#server-settings-dialog'));};
$('#server-settings-dialog').addEventListener('close',()=>{$('#server-api-key').value='';hideApiKey();});
$('#model-editor').addEventListener('close',syncModelEditorTriggers);
$('#model-search').oninput=filterModels;
$$('[data-model-filter]').forEach(button=>button.onclick=()=>{modelFilter=button.dataset.modelFilter;filterModels();});
