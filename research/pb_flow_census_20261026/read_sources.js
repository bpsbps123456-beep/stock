// Run with: Get-Content -Raw -Encoding UTF8 <this file> | ego-browser nodejs
// Public article text only; no chart or price widgets are extracted.
const fs = await import('node:fs/promises');
const root = 'C:/Users/geunt/OneDrive/Desktop/주식/research/pb_flow_census_20261026';
const task = await taskSpace(3);
const managed = await task.pages();
while ((await task.pages()).length < 3) await task.newPage();
const pages = (await task.pages()).slice(0, 3);
const sources = JSON.parse(await fs.readFile(root + '/reading_queue.json', 'utf8'));
await fs.mkdir(root + '/article_reads', {recursive:true});
const done = new Set((await fs.readdir(root + '/article_reads')).map(n=>n.replace('.json','')));
const queue = sources.filter(r=>!done.has(r.source_id)).slice(0, 24);
let cursor = 0;
await Promise.all(pages.map(async page => {
  while (cursor < queue.length) {
    const r = queue[cursor++];
    const out = {source_id:r.source_id, requested_url:r.url, read_at_utc:new Date().toISOString()};
    try {
      await page.goto(r.url);
      Object.assign(out, await page.evaluate(() => {
        const block=document.querySelector('[data-article-blocks]') || document.querySelector('[data-test-id="article-content"]') || document.querySelector('.caas-body') || document.querySelector('.article-body') || document.querySelector('article') || document.querySelector('main');
        const paragraphs=block ? [...block.querySelectorAll('p')].map(p=>p.innerText.trim()).filter(Boolean) : [];
        const relevant=paragraphs.filter(t=>/hedge|Goldman|Morgan Stanley|JPMorgan|BofA|Bank of America|prime book|long (?:pos|buy)|short (?:pos|sell|cover|bet)|leverage|funds|clients|speculators/i.test(t));
        const dates=[];
        function visit(o){if(!o||typeof o!=='object')return;for(const [k,v] of Object.entries(o)){if(/datePublished|dateModified|dateCreated/.test(k))dates.push({key:k,value:v});else if(typeof v==='object')visit(v);}}
        for(const s of document.querySelectorAll('script[type="application/ld+json"]')){try{visit(JSON.parse(s.textContent));}catch{}}
        for(const m of document.querySelectorAll('meta[property*="published_time"],meta[name="date"]'))dates.push({key:'meta',value:m.content});
        return {url:location.href,title:document.title,dates,paragraph_count:paragraphs.length,flow_paragraphs:relevant,body_found:!!block};
      }));
    } catch(e){out.error=String(e);}
    await fs.writeFile(root+'/article_reads/'+r.source_id+'.json',JSON.stringify(out,null,2),'utf8');
    console.log({source_id:r.source_id,date:out.dates?.[0]?.value,paragraphs:out.flow_paragraphs?.length,error:out.error});
  }
}));
console.log({batch:queue.length,remaining:sources.length-done.size-queue.length});
