(()=>{
  'use strict';
  const menu=document.querySelector('.menu-toggle');
  const sidebar=document.getElementById('sidebar');
  menu.addEventListener('click',()=>{const open=sidebar.classList.toggle('open');menu.setAttribute('aria-expanded',String(open));});
  sidebar.querySelectorAll('a').forEach(a=>a.addEventListener('click',()=>{sidebar.classList.remove('open');menu.setAttribute('aria-expanded','false');}));
  const links=[...sidebar.querySelectorAll('a[href^="#"]')];
  const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting)for(const a of links)a.classList.toggle('active',a.hash==='#'+entry.target.id);}},{rootMargin:'-12% 0px -65% 0px'});
  document.querySelectorAll('main>section[id]').forEach(section=>observer.observe(section));
})();
