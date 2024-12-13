async function fetchUploadedImages(sessionId) {
    const url = `/uploaded-images/${sessionId}`;

    try {
        const response = await fetch(url);

        if (!response.ok) {
            console.error(`Error ${response.status}: ${response.statusText}`);
            return;
        }

        const data = await response.json();

        if (!data || !data.images || !Array.isArray(data.images)) {
            console.error("Invalid data format received from API");
            return;
        }

        displayImages(data.session_id, data.images);
    } catch (error) {
        console.error("Error fetching uploaded images:", error);
    }
}

function displayImages(sessionId, images) {
    const imageSlider = document.getElementById("image-slider");

    if (!imageSlider) {
        console.error("Image slider element not found");
        return;
    }

    imageSlider.innerHTML = "";

    const fragment = document.createDocumentFragment();

    let isFirstImage = true;

    images.forEach(image => {
        const imgElement = document.createElement("img");
        imgElement.src = `/static/sessions/${sessionId}/${image}`;
        imgElement.className = "fade-image";
        imgElement.style.maxHeight = "17vh";
        imgElement.alt = `Image ${image.filename}`;

        // Make the first image active by default
        if (isFirstImage) {
            imgElement.classList.add("active");
            const currentImage = document.getElementById("currentImage");
            if (currentImage) {
                currentImage.src = imgElement.src;
            }
            isFirstImage = false;
        }

        fragment.appendChild(imgElement);
    });

    imageSlider.appendChild(fragment);
}

function getSessionIdFromPath() {
    const pathSegments = window.location.pathname.split('/').filter(Boolean);
    return pathSegments[pathSegments.length - 1] || null;
}

const sessionId = getSessionIdFromPath();

if (sessionId) {
    fetchUploadedImages(sessionId);
} else {
    console.error("Session ID not found in the URL");
}

document.getElementById("image-slider").addEventListener("click", event => {
    if (event.target.tagName === "IMG" && event.target.classList.contains("fade-image")) {
        document.querySelectorAll(".fade-image").forEach(img => img.classList.remove("active"));
        event.target.classList.add("active");

        const currentImage = document.getElementById("currentImage");
        if (currentImage) {
            currentImage.src = event.target.src;
        }
    }
});
