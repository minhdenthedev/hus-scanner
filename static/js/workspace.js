const currentImage = document.getElementById("currentImage");
var processingState = {};


async function fetchUploadedImages(sessionId) {
    const url = `/uploaded-images/${encodeURIComponent(sessionId)}`;

    try {
        const response = await fetch(url);

        if (!response.ok) {
            console.error(`Error ${response.status}: ${response.statusText}`);
            throw new Error(`Failed to fetch images. Status: ${response.status}`);
        }

        const data = await response.json();

        // Validate data structure
        if (!data || !data.images || !Array.isArray(data.images)) {
            console.error("Invalid data format received from API");
            throw new Error("Invalid data format");
        }

        // Initialize processing state
        data.images.forEach(image => {
            processingState[image] = {
                rotation: 0,
                cropping: {
                    topleft: 0,
                    topright: 0,
                    bottomleft: 0,
                    bottomright: 0,
                },
            };
        });

        // Display images and return processing state
        displayImages(data.session_id, data.images);
    } catch (error) {
        console.error("Error fetching uploaded images:", error);
        throw error; // Ensure the caller can handle the error
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
        imgElement.setAttribute("data-image-id", image)

        // Make the first image active by default
        if (isFirstImage) {
            imgElement.classList.add("active");
            if (currentImage) {
                currentImage.src = imgElement.src;
                currentImage.setAttribute("data-image-id", imgElement.getAttribute("data-image-id"));                    
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

        if (currentImage) {
            currentImage.src = event.target.src;
            image_id = event.target.getAttribute("data-image-id");
            currentImage.setAttribute("data-image-id", image_id);
            currentImage.style.transform = `rotate(${processingState[image_id].rotation}deg)`;
        }
    }
});

function rotateLeft() {
    image_id = currentImage.getAttribute("data-image-id");
    processingState[image_id].rotation -= 90;
    processingState[image_id].rotation %= 360;
    currentImage.style.transform = `rotate(${processingState[image_id].rotation}deg)`; 

    const originalWidth = currentImage.naturalWidth; // The original width of the image
    const originalHeight = currentImage.naturalHeight; // The original height of the image

    // Calculate the scale factor to make the height equal to the original width
    const scaleFactor = originalWidth / originalHeight;
    currentImage.style.scale = scaleFactor;
}

function rotateRight() {
    image_id = currentImage.getAttribute("data-image-id");
    processingState[image_id].rotation += 90;
    processingState[image_id].rotation %= 360;
    currentImage.style.transform = `rotate(${processingState[image_id].rotation}deg)`;
}