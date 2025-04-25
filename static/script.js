let scene, camera, renderer, sphere;

function init3DChatbot() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ alpha: true });
    renderer.setSize(200, 200);
    document.getElementById('chatbot-3d').appendChild(renderer.domElement);

    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const material = new THREE.MeshPhongMaterial({ color: 0x1f77b4, shininess: 100 });
    sphere = new THREE.Mesh(geometry, material);
    scene.add(sphere);

    const light = new THREE.PointLight(0xffffff, 1, 100);
    light.position.set(5, 5, 5);
    scene.add(light);

    camera.position.z = 3;

    function animate() {
        requestAnimationFrame(animate);
        sphere.rotation.y += 0.01;
        renderer.render(scene, camera);
    }
    animate();
}

async function uploadPDF() {
    const files = document.getElementById('pdf-upload').files;
    if (files.length === 0) {
        alert('Please select at least one PDF file.');
        return;
    }

    const formData = new FormData();
    for (let file of files) {
        formData.append('files', file);
    }

    try {
        const response = await axios.post('/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        alert(response.data.message);
    } catch (error) {
        alert('Error uploading PDF(s): ' + (error.response?.data?.error || 'Unknown error'));
    }
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const sessionId = document.getElementById('session-id').value || 'default_session';
    const message = input.value.trim();

    if (!message) return;

    // Display user message
    const chatContainer = document.getElementById('chat-container');
    const userMessage = document.createElement('div');
    userMessage.className = 'user-message';
    userMessage.textContent = message;
    chatContainer.appendChild(userMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    input.value = '';

    try {
        const response = await axios.post('/chat', {
            input: message,
            session_id: sessionId
        });

        // Display assistant message
        const assistantMessage = document.createElement('div');
        assistantMessage.className = 'assistant-message';
        assistantMessage.textContent = response.data.answer;
        chatContainer.appendChild(assistantMessage);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    } catch (error) {
        alert('Error: ' + (error.response?.data?.error || 'Unknown error'));
    }
}

async function clearHistory() {
    const sessionId = document.getElementById('session-id').value || 'default_session';
    try {
        const response = await axios.post('/clear_history', { session_id: sessionId });
        document.getElementById('chat-container').innerHTML = '';
        alert(response.data.message);
    } catch (error) {
        alert('Error clearing history: ' + (error.response?.data?.error || 'Unknown error'));
    }
}

// Initialize 3D chatbot on page load
window.onload = init3DChatbot;

// Send message on Enter key
document.getElementById('user-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});