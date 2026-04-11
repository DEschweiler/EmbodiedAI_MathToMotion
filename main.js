const state = {
  posts: [],
  filtered: [],
  active: null,
  search: ''
};

const CONTENT_DIR = 'sessions';

const elements = {
  postList: document.getElementById('post-list'),
  postTitle: document.getElementById('post-title'),
  postSummary: document.getElementById('post-summary'),
  postContent: document.getElementById('post-content'),
  tagList: document.getElementById('tag-list'),
  search: document.getElementById('search'),
  prev: document.getElementById('prev-post'),
  next: document.getElementById('next-post'),
  reader: document.querySelector('.reader')
};

let mathReady;
function ensureMathReady() {
  if (mathReady) return mathReady;
  mathReady = new Promise(resolve => {
    const ensureScript = () => {
      if (!document.getElementById('mathjax-script')) {
        const s = document.createElement('script');
        s.id = 'mathjax-script';
        s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
        document.head.appendChild(s);
      }
    };
    ensureScript();
    const check = () => {
      if (window.MathJax && typeof MathJax.typesetPromise === 'function') {
        if (MathJax.startup && MathJax.startup.promise) {
          MathJax.startup.promise.then(() => resolve(MathJax));
        } else {
          resolve(MathJax);
        }
      } else {
        setTimeout(check, 50);
      }
    };
    check();
  });
  return mathReady;
}

async function typesetContent() {
  try {
    await ensureMathReady();
    await MathJax.typesetPromise([elements.postContent]);
  } catch (err) {
    console.error('MathJax typeset error', err);
  }
}

async function init() {
  try {
    const manifest = await fetchJson(`${CONTENT_DIR}/index.json`);
    const entries = manifest.sessions || manifest.posts || [];
    const metas = await Promise.all(entries.map(loadFrontmatter));
    state.posts = metas
      .filter(Boolean)
      .sort((a, b) => {
        const aNum = Number.isFinite(a.idNumber) ? a.idNumber : Number.MAX_SAFE_INTEGER;
        const bNum = Number.isFinite(b.idNumber) ? b.idNumber : Number.MAX_SAFE_INTEGER;
        if (aNum !== bNum) return aNum - bNum;
        return String(a.id).localeCompare(String(b.id));
      });
    applyFilters();
    const initial = state.filtered[0];
    if (initial) {
      selectPost(initial.id);
    } else {
      elements.postContent.textContent = 'No sessions found.';
    }
    elements.search.addEventListener('input', onSearch);
    elements.prev.addEventListener('click', () => selectAdjacent(-1));
    elements.next.addEventListener('click', () => selectAdjacent(1));
  } catch (err) {
    console.error(err);
    elements.postContent.textContent = `Failed to load sessions: ${err.message}. If opened via file://, please run a local server or deploy to a static host.`;
  }
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json();
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.text();
}

async function loadFrontmatter(entry) {
  try {
    const raw = await fetchText(`${CONTENT_DIR}/${entry.file}`);
    const parsed = parseFrontmatter(raw);
    const idRaw = parsed.meta && parsed.meta.id !== undefined ? parsed.meta.id : null;
    const idNumber = Number.isFinite(Number(idRaw)) ? Number(idRaw) : null;
    const id = idRaw !== null && idRaw !== undefined ? String(idRaw) : '';
    return {
      ...parsed.meta,
      id,
      idNumber,
      file: entry.file,
      content: parsed.content
    };
  } catch (err) {
    console.error('Error loading', entry.file, err);
    return null;
  }
}

function parseFrontmatter(raw) {
  if (raw.startsWith('---')) {
    const end = raw.indexOf('---', 3);
    if (end === -1) {
      return { meta: {}, content: raw };
    }
    const yamlText = raw.slice(3, end).trim();
    const meta = jsyaml.load(yamlText) || {};
    const content = raw.slice(end + 3).trim();
    return { meta, content };
  }
  return { meta: {}, content: raw };
}

function onSearch(e) {
  state.search = e.target.value.toLowerCase();
  applyFilters();
}

function applyFilters() {
  state.filtered = state.posts.filter(p => {
    const haystack = [p.title, p.summary, (p.tags || []).join(' '), (p.learning_goals || []).join(' ')].join(' ').toLowerCase();
    const matchesSearch = haystack.includes(state.search);
    return matchesSearch;
  });
  renderPostList();
}

function renderPostList() {
  elements.postList.innerHTML = '';
  state.filtered.forEach(p => {
    const item = document.createElement('div');
    item.className = 'post-item' + (state.active === p.id ? ' active' : '');
    item.innerHTML = `
      <div class="title">${p.title || p.file}</div>
      <div class="meta"><span>Section ${p.id ?? ''}</span></div>
    `;
    item.addEventListener('click', () => selectPost(p.id));
    elements.postList.appendChild(item);
  });
}

async function selectPost(id) {
  const post = state.posts.find(p => p.id === id);
  if (!post) return;
  state.active = id;
  renderPostList();
  await renderPost(post);
}

function selectAdjacent(direction) {
  if (!state.active) return;
  const idx = state.filtered.findIndex(p => p.id === state.active);
  const nextIdx = idx + direction;
  if (nextIdx >= 0 && nextIdx < state.filtered.length) {
    selectPost(state.filtered[nextIdx].id);
  }
}

async function renderPost(post) {
  elements.postTitle.textContent = post.title || post.file;
  elements.postSummary.textContent = post.summary || '';
  elements.tagList.innerHTML = '';
  (post.tags || []).forEach(tag => {
    const pill = document.createElement('span');
    pill.className = 'tag';
    pill.textContent = tag;
    elements.tagList.appendChild(pill);
  });

  const currentIdx = state.filtered.findIndex(p => p.id === post.id);
  elements.prev.disabled = currentIdx <= 0;
  elements.next.disabled = currentIdx === -1 || currentIdx >= state.filtered.length - 1;

  const raw = await fetchText(`${CONTENT_DIR}/${post.file}`);
  const parsed = parseFrontmatter(raw);
  const html = marked.parse(parsed.content, { mangle: false, headerIds: false });
  elements.postContent.innerHTML = html;
  elements.postContent.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
  if (elements.reader) {
    elements.reader.scrollTo({ top: 0, behavior: 'smooth' });
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
  typesetContent();
}

document.addEventListener('DOMContentLoaded', init);
