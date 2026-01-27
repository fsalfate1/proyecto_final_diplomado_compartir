import { sendMessage, sendMessageStream } from './api.js';
import { getStoredUser, saveStoredUser, getCurrentLocation } from './utils.js';

const chatContainer = document.getElementById('chat-container');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const quickActions = document.getElementById('quick-actions');
const onboardingOverlay = document.getElementById('onboarding-overlay');
const phoneInput = document.getElementById('phone-input');
const startBtn = document.getElementById('start-btn');
const onboardingError = document.getElementById('onboarding-error');
const termsOverlay = document.getElementById('terms-overlay');
const acceptTermsBtn = document.getElementById('accept-terms-btn');

// Location Elements
const locationBtn = document.getElementById('location-btn');
const historyBtn = document.getElementById('history-btn');
const locationStatus = document.getElementById('location-status');
const mapModal = document.getElementById('map-modal');
const closeMapBtn = document.getElementById('close-map-btn');
const confirmLocationBtn = document.getElementById('confirm-location-btn');

let isWaiting = false;
let currentCoords = null; // { lat, lon, accuracy }
let mapInstance = null;
let mapMarker = null;
let messageSeq = 0;
let pendingPersist = null;

function nextId(prefix) {
  messageSeq += 1;
  const rand = (globalThis.crypto && crypto.randomUUID)
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);
  const base = `${prefix}-${Date.now()}-${messageSeq}-${rand}`;
  return base;
}

function isHistoryQuery(text) {
  const normalized = (text || '').toLowerCase();
  const keywords = [
    'historial',
    'ficha medica',
    'ficha médica',
    'resumen',
    'historia clinica',
    'historia clínica',
    'mi ficha',
    'mi historial',
    'resumen de mi ficha',
    'resumen de mi ficha medica',
    'resumen de mi ficha médica'
  ];
  return keywords.some(k => normalized.includes(k));
}

// --- Initialization ---

function init() {
  const user = getStoredUser();
  if (!user) {
    termsOverlay.style.display = 'flex';
  } else {
    showWelcomeMessage(user.telefono);
    // Try to get location silently on startup
    attemptAutoLocation();
  }

  // Auto-resize textarea
  chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
  });

  // Enter to send (Shift+Enter for new line)
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  sendBtn.addEventListener('click', handleSend);
  startBtn.addEventListener('click', handleOnboarding);
  acceptTermsBtn.addEventListener('click', () => {
    termsOverlay.style.display = 'none';
    onboardingOverlay.style.display = 'flex';
  });

  // Map / Location Events
  locationBtn.addEventListener('click', openMapModal);
  historyBtn.addEventListener('click', () => {
    handleSend("Resumen de mi ficha médica 📁");
  });
  closeMapBtn.addEventListener('click', closeMapModal);
  confirmLocationBtn.addEventListener('click', confirmLocation);
  
  // Close modal on click outside
  mapModal.addEventListener('click', (e) => {
    if (e.target === mapModal) closeMapModal();
  });
}

// --- Location & Map Logic ---

async function attemptAutoLocation() {
  try {
    const loc = await getCurrentLocation();
    updateLocationState(loc);
  } catch (e) {
    console.log("Auto-location failed or denied:", e);
    locationStatus.textContent = "Sin ubicación";
  }
}

function updateLocationState(loc) {
  currentCoords = loc;
  locationStatus.textContent = `Lat: ${loc.lat.toFixed(4)}`;
  locationBtn.classList.add('active'); // Optional styling hook
}

function openMapModal() {
  mapModal.style.display = 'flex';
  
  // Wait for modal to be visible before sizing map
  setTimeout(() => {
    initMap();
  }, 100);
}

function closeMapModal() {
  mapModal.style.display = 'none';
}

function initMap() {
  // Default to Santiago center if no coords yet
  const defaultLat = -33.4489;
  const defaultLon = -70.6693;
  
  const lat = currentCoords ? currentCoords.lat : defaultLat;
  const lon = currentCoords ? currentCoords.lon : defaultLon;

  if (!mapInstance) {
    mapInstance = L.map('map').setView([lat, lon], 13);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(mapInstance);

    mapMarker = L.marker([lat, lon], { draggable: true }).addTo(mapInstance);
    
    mapMarker.on('dragend', function(e) {
      // Update marker position internal tracking if needed
      // We read from marker on confirm
    });
    
    // Map click moves marker
    mapInstance.on('click', function(e) {
      mapMarker.setLatLng(e.latlng);
    });

  } else {
    mapInstance.invalidateSize();
    mapInstance.setView([lat, lon], 13);
    mapMarker.setLatLng([lat, lon]);
  }
}

function confirmLocation() {
  if (!mapMarker) return;
  
  const { lat, lng } = mapMarker.getLatLng();
  updateLocationState({
    lat: lat,
    lon: lng,
    accuracy: 10 // Mock accuracy for manual selection
  });
  
  closeMapModal();
}

// --- Onboarding ---

function handleOnboarding() {
  const phone = phoneInput.value.trim();
  // Basic validation for Chile phone number (lenient)
  if (phone.length < 8) {
    onboardingError.textContent = 'Por favor ingresa un número válido.';
    onboardingError.style.display = 'block';
    return;
  }
  saveStoredUser(phone);
  onboardingOverlay.style.display = 'none';
  showWelcomeMessage(phone);
  attemptAutoLocation();
}

function showWelcomeMessage(phone) {
  if (chatContainer.children.length === 0) {
    addMessage('bot', "Hola 👋 Soy Pulso. Te ayudo a entender tus recetas médicas, recordar tus remedios, y encontrar farmacias cuando las necesitas.\n\n¿En qué puedo ayudarte hoy?");
    showQuickActions(['💊 Consultar medicamento', '🗺️ Farmacias de turno', '🩺 Tengo un síntoma']);
  }
}

// --- Messaging ---

async function handleSend(overrideText = null) {
  const text = overrideText || chatInput.value.trim();
  if (!text || isWaiting) return;

  // UI Updates
  addMessage('user', text);
  if (!overrideText) {
    chatInput.value = '';
    chatInput.style.height = 'auto';
  }
  hideQuickActions();
  isWaiting = true;
  
  const loadingId = showTyping();

  try {
    if (isHistoryQuery(text) && pendingPersist) {
      try {
        await pendingPersist;
      } catch (e) {
        console.warn('Pending persist failed:', e);
      }
    }
    // 1. Ensure we have location if possible. 
    // If user hasn't set it manually (currentCoords is null), try auto one last time.
    let locToSend = currentCoords;
    
    if (!locToSend) {
      try {
        locToSend = await getCurrentLocation();
        updateLocationState(locToSend); // Save it for future
      } catch (err) {
        console.warn("Location access denied or error:", err);
        // Fallback: Send null/null to backend
        locToSend = { lat: null, lon: null, accuracy: null };
      }
    }

    // 2. Prepare Payload
    const user = getStoredUser();
    const payload = {
      telefono: user.telefono,
      message: text,
      lat: locToSend.lat,
      lon: locToSend.lon,
      accuracy: locToSend.accuracy
    };

    // 3. Send
    removeMessage(loadingId);
    const botId = addMessage('bot', '');
    let streamedText = '';

    try {
      const controller = new AbortController();
      let gotFirstChunk = false;
      const streamTimeout = setTimeout(() => {
        if (!gotFirstChunk) controller.abort();
      }, 15000);

      await sendMessageStream(payload, {
        signal: controller.signal,
        onChunk: (chunk) => {
          gotFirstChunk = true;
          clearTimeout(streamTimeout);
          streamedText += chunk;
          updateMessageText(botId, streamedText);
        },
        onMeta: (raw) => {
          gotFirstChunk = true;
          clearTimeout(streamTimeout);
          let data = null;
          try {
            data = JSON.parse(raw);
          } catch (e) {
            console.error('Meta JSON parse error:', e);
          }
          if (!data) return;

          const extras = [];
          if (data.farmacias_abiertas && data.farmacias_abiertas.length > 0) {
            extras.push(renderPharmacies(data.farmacias_abiertas, "Farmacias de Turnos"));
          }
          if (data.farmacias && data.farmacias.length > 0) {
            extras.push(renderPharmacies(data.farmacias, "Farmacias Cercanas"));
          }
          if (extras.length > 0) {
            appendExtrasToMessage(botId, extras);
          }
        }
      });
      // Persist history in Neon using the non-stream endpoint (no UI update).
      pendingPersist = sendMessage({ ...payload, persist_only: true })
        .catch((e) => {
          console.warn('Persist-only request failed:', e);
        })
        .finally(() => {
          pendingPersist = null;
        });
      clearTimeout(streamTimeout);
    } catch (streamErr) {
      console.warn('Streaming failed, fallback to JSON:', streamErr);
      const data = await sendMessage(payload);
      const botText = data.answer || "Lo siento, no pude procesar tu solicitud.";
      updateMessageText(botId, botText);
      const extras = [];
      if (data.farmacias_abiertas && data.farmacias_abiertas.length > 0) {
        extras.push(renderPharmacies(data.farmacias_abiertas, "Farmacias de Turnos"));
      }
      if (data.farmacias && data.farmacias.length > 0) {
        extras.push(renderPharmacies(data.farmacias, "Farmacias Cercanas"));
      }
      if (extras.length > 0) {
        appendExtrasToMessage(botId, extras);
      }
    }
    
  } catch (err) {
    removeMessage(loadingId);
    addMessage('bot', 'Ocurrió un error al conectar con el servidor. Por favor intenta de nuevo.');
    console.error(err);
  } finally {
    isWaiting = false;
  }
}

function renderPharmacies(list, title) {
    let html = `<div style="margin-top: 1rem;"><h4 style="margin-bottom:0.5rem; color:var(--pulso-deep);">${title}</h4>`;
    list.forEach(f => {
        html += `
        <div class="info-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <strong>${f.nombre}</strong>
                <span class="meta">${f.distancia_km} km</span>
            </div>
            <p style="margin:0.2rem 0;">${f.direccion}, ${f.comuna}</p>
            ${f.apertura ? `<p style="font-size:0.8rem; color:var(--pulso-heart);">Abierto: ${f.apertura} - ${f.cierre}</p>` : ''}
        </div>`;
    });
    html += "</div>";
    return html;
}

// --- UI Helpers ---

function addMessage(sender, text, extrasHTML = []) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${sender}`;
  let id = nextId('msg');
  while (document.getElementById(id)) {
    id = nextId('msg');
  }
  msgDiv.id = id;
  
  // Format text (convert newlines to br)
  const formattedText = text.replace(/\n/g, '<br>');
  
  let content = `<p>${formattedText}</p>`;
  
  if (extrasHTML.length > 0) {
    content += extrasHTML.join('');
  }

  msgDiv.innerHTML = content;
  chatContainer.appendChild(msgDiv);
  scrollToBottom();
  return msgDiv.id;
}

function updateMessageText(messageId, text) {
  const el = document.getElementById(messageId);
  if (!el) return;
  let p = el.querySelector('p');
  if (!p) {
    p = document.createElement('p');
    el.prepend(p);
  }
  p.dataset.raw = text;
  p.innerHTML = text.replace(/\n/g, '<br>');
  scrollToBottom();
}

function appendExtrasToMessage(messageId, extrasHTML = []) {
  const el = document.getElementById(messageId);
  if (!el || extrasHTML.length === 0) return;
  el.insertAdjacentHTML('beforeend', extrasHTML.join(''));
  scrollToBottom();
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function showTyping() {
  const id = nextId('typing');
  const msgDiv = document.createElement('div');
  msgDiv.id = id;
  msgDiv.className = 'message bot';
  msgDiv.innerHTML = `
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  chatContainer.appendChild(msgDiv);
  scrollToBottom();
  return id;
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showQuickActions(actions) {
  quickActions.innerHTML = '';
  actions.forEach(action => {
    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.textContent = action;
    chip.addEventListener('click', () => {
      chatInput.value = action;
      handleSend(); // Send immediately
    });
    quickActions.appendChild(chip);
  });
  quickActions.style.display = 'flex';
}

function hideQuickActions() {
  quickActions.style.display = 'none';
}

// Run
init();
