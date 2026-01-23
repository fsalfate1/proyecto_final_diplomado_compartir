import { sendMessage } from './api.js';
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
    const data = await sendMessage(payload);

    // 4. Remove typing, show response
    removeMessage(loadingId);
    
    // Format response
    let botText = data.answer || "Lo siento, no pude procesar tu solicitud.";
    
    // Append structured data if available
    const extras = [];
    
    // Farmacias
    if (data.farmacias_abiertas && data.farmacias_abiertas.length > 0) {
        extras.push(renderPharmacies(data.farmacias_abiertas, "Farmacias Abiertas"));
    } 
    if (data.farmacias && data.farmacias.length > 0) {
        extras.push(renderPharmacies(data.farmacias, "Farmacias Cercanas"));
    }

    addMessage('bot', botText, extras);
    
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
  
  // Format text (convert newlines to br)
  const formattedText = text.replace(/\n/g, '<br>');
  
  let content = `<p>${formattedText}</p>`;
  
  if (extrasHTML.length > 0) {
    content += extrasHTML.join('');
  }

  msgDiv.innerHTML = content;
  chatContainer.appendChild(msgDiv);
  scrollToBottom();
  return msgDiv.id = 'msg-' + Date.now();
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function showTyping() {
  const id = 'typing-' + Date.now();
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