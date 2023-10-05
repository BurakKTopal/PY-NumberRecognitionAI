document.addEventListener("DOMContentLoaded", function () {
// Get a reference to your canvas element
var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");
ctx.fillStyle = "white";

// Variables to track the current and previous mouse coordinates
var prevX, prevY;
var currX, currY;

// Flag to determine whether to start drawing
var isDrawing = false;

// Set the line color and width
ctx.strokeStyle = "black"; // Change this to your desired color
ctx.lineWidth = 15; // Change this to your desired line width

// Add an event listener to track mouse movements
canvas.addEventListener("mousemove", function (event) {
    if (isDrawing) {
        // Capture the current mouse coordinates
        currX = event.clientX - canvas.getBoundingClientRect().left;
        currY = event.clientY - canvas.getBoundingClientRect().top;

        // Draw a line from the previous point to the current point
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(currX, currY);
        ctx.stroke();

        // Update the previous coordinates
        prevX = currX;
        prevY = currY;
    }
});

// Add an event listener to start drawing when the mouse button is pressed
canvas.addEventListener("mousedown", function (event) {
    // Capture the starting mouse coordinates
    prevX = event.clientX - canvas.getBoundingClientRect().left;
    prevY = event.clientY - canvas.getBoundingClientRect().top;
    isDrawing = true;
});

// Add an event listener to stop drawing when the mouse button is released
canvas.addEventListener("mouseup", function () {
    isDrawing = false;
});

// Function to clear the canvas
 document.getElementById("clearButton").addEventListener("click", function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
})
// Function to save the canvas as an image (optional)


document.getElementById("saveButton").addEventListener("click", function () {
    // Create a temporary canvas to include the white background
    var tempCanvas = document.createElement("canvas");
    var tempCtx = tempCanvas.getContext("2d");
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;

    // Draw a white background on the temporary canvas
    tempCtx.fillStyle = "white";
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    // Draw the content of the original canvas on the temporary canvas
    tempCtx.drawImage(canvas, 0, 0);

    // Convert the temporary canvas to an image
    var image = tempCanvas.toDataURL("image/png");

    // Send the image data to the server using an AJAX request or form submission
    // Example AJAX request using fetch:
    fetch('/upload_canvas', {
        method: 'POST',
        body: JSON.stringify({ image: image }),
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(function (response) {
        window.location.href = '/output';
    });
});



});
