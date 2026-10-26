'use strict';
fetch('/data/reports.json').then(response=>{
  if(!response.ok) throw new Error('无法读取报告目录');
  return response.json();
}).then(data=>{
  const container=document.querySelector('#reports');
  container.replaceChildren();
  const reports=Array.isArray(data)?data:(data.reports||[]);
  if(!reports.length){container.textContent='尚未添加报告。';return;}
  for(const report of reports){
    const article=document.createElement('article');
    const heading=document.createElement('h3');
    const link=document.createElement('a');
    const url=new URL(report.href||report.url||'/',window.location.href);
    if(!['http:','https:'].includes(url.protocol)) throw new Error('无效的报告地址');
    link.href=url.href;link.textContent=report.title||report.id||'实验报告';
    heading.append(link);article.append(heading);
    if(report.summary){const p=document.createElement('p');p.textContent=report.summary;article.append(p);}
    container.append(article);
  }
}).catch(error=>{document.querySelector('#reports').textContent=error.message;});
