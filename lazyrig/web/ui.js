'use strict';
window.UI=(()=>{
const shapes={"grid":"<rect x=\"3\" y=\"3\" width=\"7\" height=\"7\" rx=\"2\"/><rect x=\"14\" y=\"3\" width=\"7\" height=\"7\" rx=\"2\"/><rect x=\"3\" y=\"14\" width=\"7\" height=\"7\" rx=\"2\"/><rect x=\"14\" y=\"14\" width=\"7\" height=\"7\" rx=\"2\"/>","chip":"<rect x=\"6\" y=\"6\" width=\"12\" height=\"12\" rx=\"3\"/><rect x=\"9\" y=\"9\" width=\"6\" height=\"6\" rx=\"1\"/><path d=\"M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4\"/>","gpu":"<rect x=\"3\" y=\"6\" width=\"18\" height=\"12\" rx=\"2\"/><circle cx=\"10\" cy=\"12\" r=\"3\"/><path d=\"M16 9h2m-2 3h2m-2 3h2M6 18v3m3-3v3M3 9H1\"/>","code":"<path d=\"m8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18\"/>","import":"<path d=\"M12 3v12m-4-4 4 4 4-4M4 16v4h16v-4\"/>","chat":"<path d=\"M21 12a8 8 0 0 1-8 8H4l-2 2V11a8 8 0 0 1 8-8h3a8 8 0 0 1 8 9Z\"/><path d=\"M7 9h9m-9 5h6\"/>","gear":"<path d=\"m10 2-.7 3-2 .9L4.7 5 2 9.5l2.2 2.1v2.2L2 16l2.7 4.5 2.6-.9 2 .9.7 2.5h4l.7-2.5 2-.9 2.6.9L22 16l-2.2-2.2v-2.2L22 9.5 19.3 5l-2.6.9-2-.9L14 2Z\" transform=\"translate(1.1 .5) scale(.91)\"/><circle cx=\"12\" cy=\"12\" r=\"3\"/>","check":"<path d=\"m5 12 4 4L19 6\"/>","x":"<path d=\"m6 6 12 12M6 18 18 6\"/>","plus":"<path d=\"M12 5v14M5 12h14\"/>","play":"<path d=\"m8 5 11 7-11 7Z\"/>","stop":"<rect x=\"6\" y=\"6\" width=\"12\" height=\"12\" rx=\"2\"/>","folder":"<path d=\"M3 7V5a2 2 0 0 1 2-2h5l3 4h6a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z\"/><path d=\"M3 8h18\"/>","folder-open":"<path d=\"M3 17V5a2 2 0 0 1 2-2h5l3 4h6a2 2 0 0 1 2 2v2M4 11h18l-3 9H2Z\"/>","search":"<circle cx=\"10.5\" cy=\"10.5\" r=\"6.5\"/><path d=\"m16 16 5 5\"/>","right":"<path d=\"m9 5 7 7-7 7\"/>","down":"<path d=\"m6 9 6 6 6-6\"/>","refresh":"<path d=\"M20 7a8 8 0 1 0 0 10M20 3v5h-5\"/>","copy":"<rect x=\"8\" y=\"8\" width=\"13\" height=\"13\" rx=\"3\"/><path d=\"M15 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h3\"/>","spark":"<path d=\"m12 2 2.7 7.3L22 12l-7.3 2.7L12 22l-2.7-7.3L2 12l7.3-2.7Z\"/>","image":"<rect x=\"3\" y=\"3\" width=\"18\" height=\"18\" rx=\"4\"/><circle cx=\"8\" cy=\"8\" r=\"1.5\"/><path d=\"m3 16 5-5 5 5 3-3 5 5\"/>","pulse":"<path d=\"M2 12h5l3-8 4 16 3-8h5\"/>","terminal":"<rect x=\"2\" y=\"4\" width=\"20\" height=\"16\" rx=\"4\"/><path d=\"m6 9 3 3-3 3m7 0h4\"/>","info":"<circle cx=\"12\" cy=\"12\" r=\"9\"/><path d=\"M12 11v6m0-10v.1\"/>","tune":"<path d=\"M4 5h7m5 0h4M4 12h2m5 0h9M4 19h10m5 0h1\"/><circle cx=\"13.5\" cy=\"5\" r=\"2.5\"/><circle cx=\"8.5\" cy=\"12\" r=\"2.5\"/><circle cx=\"16.5\" cy=\"19\" r=\"2.5\"/>","send":"<path d=\"M12 19V5m-6 6 6-6 6 6\"/>","history":"<rect x=\"3\" y=\"3\" width=\"18\" height=\"18\" rx=\"4\"/><path d=\"M9 3v18M13 8h4m-4 4h4m-4 4h2\"/>","trash":"<path d=\"M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7\"/>","file":"<path d=\"M14 3H5v18h14V8l-5-5Zm0 0v5h5M8 12h8m-8 4h6\"/>","bolt":"<path d=\"m13 2-9 12h7l-1 8 10-13h-7Z\"/>","shield":"<path d=\"m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6Z\"/><path d=\"m8 12 3 3 5-5\"/>","key":"<circle cx=\"8\" cy=\"8\" r=\"5\"/><path d=\"m12 12 9 9m-3-3 3-3m-6 0 3-3\"/>"};
Object.assign(shapes,{
  menu:'<path d="M5 6h14M5 12h14M5 18h14"/>',
  expand:'<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M8 8 3 3m13 5 5-5M8 16l-5 5m13-5 5 5"/>',
  more:'<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
  'folder-plus':'<path d="M3 7V5h6l3 3h9v12H3Z M12 11v6m-3-3h6"/>'
});

const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const icon=name=>'<svg class="ui-icon" viewBox="0 0 24 24" aria-hidden="true">'+(shapes[name]||shapes.gear)+'</svg>';
function button(name,label,attributes='',classes=''){
  return '<button type="button" class="icon-button '+classes+'" aria-label="'+escape(label)+'" title="'+escape(label)+'" '+attributes+'>'+icon(name)+'</button>';
}
function decorate(root=document){
  for(const el of root.querySelectorAll('[data-ui-icon]:not([data-icon-ready])')){
    const name=el.dataset.uiIcon;el.dataset.iconReady='';
    if(el.classList.contains('icon-button')){if(!el.getAttribute('aria-label'))el.setAttribute('aria-label',el.textContent.trim());el.innerHTML=icon(name);}
    else el.insertAdjacentHTML('afterbegin',icon(name));
  }
}
function open(dialog){window.Help?.hide();decorate(dialog);if(!dialog.open)dialog.showModal();}
function close(dialog){window.Help?.hide();for(const picker of dialog.querySelectorAll('[data-device-picker][open]'))picker.open=false;dialog.close();}
function reveal(control){
  const form=control.closest('form'),section=control.closest('[data-settings-section]');
  if(section)form?._settingsTab?.(section.dataset.settingsSection);
  for(let el=control.parentElement;el&&el!==form;el=el.parentElement)if(el.tagName==='DETAILS')el.open=true;
}
function tabs(form,initial='general'){
  const buttons=[...form.querySelectorAll('[data-settings-tab]')],sections=[...form.querySelectorAll('[data-settings-section]')];
  const select=(name,focus=false)=>{
    window.Help?.hide();for(const picker of form.querySelectorAll('[data-device-picker][open]'))picker.open=false;
    for(const b of buttons){const active=b.dataset.settingsTab===name;b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;if(active&&focus)b.focus();}
    for(const s of sections)s.hidden=s.dataset.settingsSection!==name;
    form.dataset.activeTab=name;const scroll=form.querySelector('.settings-scroll');if(scroll)scroll.scrollTop=0;
  };
  form._settingsTab=select;
  for(const b of buttons){b.onclick=()=>select(b.dataset.settingsTab);b.onkeydown=e=>{
    if(!['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;
    e.preventDefault();const i=buttons.indexOf(b),next=e.key==='Home'?0:e.key==='End'?buttons.length-1:(i+(['ArrowDown','ArrowRight'].includes(e.key)?1:-1)+buttons.length)%buttons.length;
    select(buttons[next].dataset.settingsTab,true);buttons[next].scrollIntoView({block:'nearest',inline:'nearest'});
  };}
  form.addEventListener('invalid',e=>reveal(e.target),true);select(initial);
}
function tabNavigation(){return '<div class="settings-tabs" role="tablist" aria-label="Model settings sections">'+[['general','grid','General'],['devices','chip','Devices'],['generation','spark','Generation'],['runtime','bolt','Runtime'],['advanced','code','Advanced']].map(([id,glyph,label])=>'<button type="button" role="tab" id="model-tab-'+id+'" data-settings-tab="'+id+'" aria-controls="model-section-'+id+'">'+icon(glyph)+label+'</button>').join('')+'</div>';}
function section(id){return '<section role="tabpanel" id="model-section-'+id+'" data-settings-section="'+id+'" aria-labelledby="model-tab-'+id+'">';}
function parseJSON(form,name){const field=form.elements[name];try{return JSON.parse(field.value);}catch(error){reveal(field);field.focus();throw new Error('Invalid JSON in '+(field.closest('.field')?.querySelector('.field-label')?.textContent||name.replaceAll('_',' '))+': '+error.message);}}
// Dialog content, alerts and focus stay in the browser's top layer.
document.addEventListener('click',e=>{const b=e.target.closest('[data-modal-close]');if(b)close(b.closest('dialog'));});
document.addEventListener('cancel',e=>{if(e.defaultPrevented)return;window.Help?.hide();},true);
document.addEventListener('DOMContentLoaded',()=>{
  decorate();
  for(const dialog of document.querySelectorAll('dialog')){
    let outside=false;
    dialog.addEventListener('pointerdown',e=>{outside=e.target===dialog;});
    dialog.addEventListener('click',e=>{if(outside&&e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)close(dialog);}outside=false;});
  }
});
return {icon,button,decorate,open,close,tabs,tabNavigation,section,reveal,parseJSON};
})();
