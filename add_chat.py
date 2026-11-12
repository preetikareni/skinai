import os, shutil

PATH = os.path.join("templates", "index.html")

CSS = r"""
/* ===== SkinAI chatbot ===== */
.chat-section{margin-top:38px;}
.chat-card{padding:24px 26px;background:var(--cream);border:1px solid var(--line);border-radius:26px;}
.chat-chips{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px;}
.chat-chip{padding:8px 16px;border-radius:999px;border:1px solid var(--line);background:#f6e9e4;color:var(--plum);font:400 13.5px var(--sans);cursor:pointer;transition:background .3s,color .3s;}
.chat-chip:hover{background:var(--plum);color:var(--cream);}
.chat-messages{display:flex;flex-direction:column;gap:10px;max-height:380px;min-height:120px;overflow-y:auto;padding:4px 2px 12px;}
.chat-msg{max-width:82%;padding:12px 16px;border-radius:18px;font-size:14.5px;line-height:1.6;white-space:pre-wrap;word-wrap:break-word;}
.chat-msg.user{align-self:flex-end;background:var(--plum);color:var(--cream);border-bottom-right-radius:5px;}
.chat-msg.bot{align-self:flex-start;background:#f1ddd6;color:var(--plum);border-bottom-left-radius:5px;}
.chat-input-row{display:flex;gap:10px;margin-top:6px;}
.chat-input-row input{flex:1;padding:13px 18px;border-radius:999px;border:1px solid var(--line);background:#fff;font:400 14.5px var(--sans);color:var(--plum);}
.chat-input-row button{padding:12px 28px;border:none;border-radius:999px;background:linear-gradient(135deg,var(--gold),#cfa56a);color:#fff;font:500 14px var(--sans);cursor:pointer;}
.chat-input-row button:disabled{opacity:.5;cursor:not-allowed;}
.chat-note{margin-top:12px;font-size:12.5px;color:var(--muted);text-align:center;}
"""

HTML = r"""
      <div class="chat-section">
        <div class="reco-title">Ask SkinAI</div>
        <div class="chat-card">
          <div class="chat-chips">
            <button type="button" class="chat-chip" onclick="sendChat(this.textContent)">Why is my skin like this?</button>
            <button type="button" class="chat-chip" onclick="sendChat(this.textContent)">Which face wash should I use?</button>
            <button type="button" class="chat-chip" onclick="sendChat(this.textContent)">Which serum is good for me?</button>
            <button type="button" class="chat-chip" onclick="sendChat(this.textContent)">Can I use it at night?</button>
          </div>
          <div class="chat-messages" id="chatMessages" aria-live="polite"></div>
          <div class="chat-input-row">
            <input id="chatInput" type="text" maxlength="500" placeholder="Ask about your skin care..." autocomplete="off"/>
            <button type="button" id="chatSend">Send</button>
          </div>
          <p class="chat-note">AI assistant for general skin-care tips. Not medical advice.</p>
        </div>
      </div>
"""

JS = r"""
/* ===== SkinAI chatbot ===== */
let chatLabel = null, chatConf = null, chatHistory = [], chatBusy = false;

function chatAdd(role, text) {
  const box = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

async function sendChat(text) {
  text = (text || '').trim();
  if (!text || chatBusy || !chatLabel) return;
  chatBusy = true;
  const input = document.getElementById('chatInput');
  const btn = document.getElementById('chatSend');
  input.value = '';
  btn.disabled = true;
  chatAdd('user', text);
  const wait = chatAdd('bot', 'Thinking...');
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, label: chatLabel, confidence: chatConf, history: chatHistory})
    });
    const data = await res.json();
    wait.textContent = data.success ? data.reply : (data.error || 'Sorry, something went wrong.');
    if (data.success) {
      chatHistory.push({role: 'user', text: text});
      chatHistory.push({role: 'bot', text: data.reply});
      chatHistory = chatHistory.slice(-12);
    }
  } catch (err) {
    wait.textContent = 'Network error. Please try again.';
  }
  chatBusy = false;
  btn.disabled = false;
  input.focus();
}

document.getElementById('chatInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') sendChat(e.target.value);
});
document.getElementById('chatSend').addEventListener('click', () => {
  sendChat(document.getElementById('chatInput').value);
});

const _showResults = showResults;
showResults = function (data) {
  _showResults(data);
  chatLabel = data.label;
  chatConf = data.confidence;
  chatHistory = [];
  document.getElementById('chatMessages').innerHTML = '';
  chatAdd('bot', 'Hi! I see ' + cap(data.label) + ' skin in your photo. Ask me about your routine, face wash, serum or night use.');
};

const _resetAll = resetAll;
resetAll = function () {
  _resetAll();
  chatLabel = null;
  chatConf = null;
  chatHistory = [];
  document.getElementById('chatMessages').innerHTML = '';
};
"""

with open(PATH, encoding="utf-8") as f:
    html = f.read()

if 'id="chatMessages"' in html:
    raise SystemExit("Chat is already added to index.html. Nothing changed.")

anchors = {
    "</style>": html.count("</style>"),
    '<div class="reco-grid" id="recoGrid"></div>': html.count('<div class="reco-grid" id="recoGrid"></div>'),
    "</script>": html.count("</script>"),
}
for a, n in anchors.items():
    if n != 1:
        raise SystemExit(f"Could not patch: expected 1 of {a!r} but found {n}. Nothing changed.")

shutil.copy(PATH, PATH + ".bak")

html = html.replace("</style>", CSS + "</style>")
html = html.replace('<div class="reco-grid" id="recoGrid"></div>',
                    '<div class="reco-grid" id="recoGrid"></div>\n' + HTML)
html = html.replace("</script>", JS + "</script>")

with open(PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("Done. Chat added to templates/index.html (backup saved as index.html.bak)")