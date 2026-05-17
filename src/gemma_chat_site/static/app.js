const statusEl = document.getElementById('status');
const chatEl = document.getElementById('chat');
const form = document.getElementById('form');
const promptEl = document.getElementById('prompt');
const sendEl = document.getElementById('send');
const tokensEl = document.getElementById('tokens');
const tempEl = document.getElementById('temp');
const messages = [];

function add(role, text) {
  const item = document.createElement('div');
  item.className = `msg ${role}`;
  item.textContent = text;
  chatEl.appendChild(item);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setStatus(data) {
  const fields = {
    loaded: data.loaded,
    model_path: data.model_path,
    model_type: data.model_type,
    architectures: (data.architectures || []).join(', '),
    cuda_available: data.cuda_available,
    gpu_name: data.gpu_name || 'none',
    memory_allocated_mb: data.memory_allocated_mb,
    memory_reserved_mb: data.memory_reserved_mb,
    error: data.error || 'none',
  };
  statusEl.innerHTML = Object.entries(fields)
    .map(([key, value]) => `<dt>${key}</dt><dd>${value ?? 'unknown'}</dd>`)
    .join('');
}

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || 'request failed');
  return data;
}

async function refreshStatus() {
  try {
    setStatus(await api('/api/status'));
  } catch (error) {
    setStatus({loaded: false, error: error.message});
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const content = promptEl.value.trim();
  if (!content) return;
  promptEl.value = '';
  sendEl.disabled = true;
  messages.push({role: 'user', content});
  add('user', content);
  try {
    const data = await api('/api/chat', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({messages, max_new_tokens: Number(tokensEl.value), temperature: Number(tempEl.value)}),
    });
    messages.push({role: 'assistant', content: data.reply});
    add('assistant', data.reply || '[empty response]');
    setStatus(data.status);
  } catch (error) {
    add('error', error.message);
    await refreshStatus();
  } finally {
    sendEl.disabled = false;
    promptEl.focus();
  }
});

document.getElementById('refresh').onclick = refreshStatus;
refreshStatus();
add('assistant', 'Real local Gemma model loaded on server startup. Send a short prompt to verify inference.');
