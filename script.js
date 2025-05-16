document.addEventListener("DOMContentLoaded", function () {
    // Scroll-down function for arrow click
    const arrowCircle = document.querySelector(".arrow-circle");
    if (arrowCircle) {
        arrowCircle.addEventListener("click", function () {
            document.querySelector("#content").scrollIntoView({ behavior: "smooth" });
        });
    }

    // Typing Effect for "PRODUCTIFY"
    const typedTextElement = document.getElementById("typed-text");
    if (typedTextElement) {
        new Typed('#typed-text', {
            strings: ["PRODUCTIFY"],
            typeSpeed: 100,
            showCursor: false
        });
    }

    // Image Selection and Preview Logic
    const chooseImageBtn = document.getElementById("choose-image");
    const imageInput = document.getElementById("image-input");
    const previewImg = document.getElementById("preview-img");
    const fileNameText = document.getElementById("file-name");
    const previewDiv = document.getElementById("preview");
    const submitBtn = document.getElementById("submit");
    const messageBox = document.getElementById("message-box");

    // Check if all required elements exist
    if (!chooseImageBtn || !imageInput || !previewImg || !fileNameText || !previewDiv || !submitBtn || !messageBox) {
        console.error("Some required elements are missing in the HTML.");
        return;
    }

    // Create and configure the remove button
    const removeImageBtn = document.createElement("button");
    removeImageBtn.id = "remove-image";
    removeImageBtn.innerHTML = "✖";
    removeImageBtn.style.display = "none"; // Initially hidden

    removeImageBtn.addEventListener("click", function () {
        previewImg.src = "";
        previewImg.hidden = true;
        fileNameText.textContent = "";
        removeImageBtn.style.display = "none"; // Hide cross
        chooseImageBtn.style.display = "inline-block"; // Show button again
    });

    previewDiv.appendChild(removeImageBtn); // Append the remove button

    // Image selection event
    chooseImageBtn.addEventListener("click", function () {
        imageInput.click();
    });

    imageInput.addEventListener("change", function (event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImg.src = e.target.result;
                previewImg.hidden = false;
                fileNameText.textContent = `File: ${file.name}`;
                chooseImageBtn.style.display = "none"; // Hide button
                removeImageBtn.style.display = "inline"; // Show cross
            };
            reader.readAsDataURL(file);
        }
    });

    // Handle Form Submission
    submitBtn.addEventListener("click", function () {
        const description = document.getElementById("description").value;
        const file = imageInput.files[0];

        if (!description.trim() || !file) {
            showMessage("Please enter a description and select an image.", "error");
            return;
        }

        const formData = new FormData();
        formData.append("description", description);
        formData.append("images[]", file);

        fetch("http://127.0.0.1:5000/submit", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log("Success:", data);
            showMessage("Submission successful!", "success");
            // Update the generated caption text area with the returned description
            updateGeneratedCaption(data.description);
        })
        .catch(error => {
            console.error("Error:", error);
            showMessage("An error occurred while submitting.", "error");
        });
    });

    // Function to Show Messages in the UI
    function showMessage(text, type) {
        if (!messageBox) return;
        messageBox.textContent = text;
        messageBox.className = `message ${type}`;
        messageBox.style.display = "block";

        // Scroll to the message box for visibility
        messageBox.scrollIntoView({ behavior: "smooth" });

        // Hide the message after 3 seconds
        setTimeout(() => {
            messageBox.style.display = "none";
        }, 3000);
    }

    // Dark Mode Toggle (Ensure this element exists in HTML)
    const themeToggle = document.getElementById("toggle-theme");
    if (themeToggle) {
        themeToggle.addEventListener("click", function () {
            document.body.classList.toggle("dark-mode");
            this.textContent = document.body.classList.contains("dark-mode") ? "☀ Light Mode" : "🌙 Dark Mode";
        });
    }
});

// Function to update the generated caption in the text area
function updateGeneratedCaption(caption) {
    const generatedCaption = document.getElementById("generated-caption");
    if (generatedCaption) {
        generatedCaption.value = caption;
    }
}
