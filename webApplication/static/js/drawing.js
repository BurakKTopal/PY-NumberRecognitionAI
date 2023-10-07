document.addEventListener("DOMContentLoaded", function () {
// Get a reference to your canvas element

var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");
var CanvasLarger = document.getElementById("CanvasLarger");
var CanvasSmaller = document.getElementById("CanvasSmaller");
var pencilThicker = document.getElementById("pencilThicker");
var pencilFiner = document.getElementById("pencilFiner");
var changeColor = document.getElementById("changeColor");

// Set the line color and width
canvas.width = 700
ctx.strokeStyle = "black"; // Change this to your desired color
ctx.lineWidth = 12; // Change this to your desired line width
var listOfColors = ["#120307", "#00292a", "#130a2e", "#260033"];

CanvasLarger.addEventListener("click", function () {
if (canvas.width < 1500){
canvas.width += 100
ctx.lineWidth = 12;
}
})
CanvasSmaller.addEventListener("click", function (){
if (canvas.width > 500){
canvas.width -= 100
ctx.lineWidth = 12;
}
})

pencilThicker.addEventListener("click", function (){
if (ctx.lineWidth < 25){
ctx.lineWidth += 1;
}
})

pencilFiner.addEventListener("click", function (){
if (ctx.lineWidth > 5){
ctx.lineWidth -= 1;
}
})

changeColor.addEventListener("click", function (){
var index = Math.floor(Math.random() * listOfColors.length)
ctx.strokeStyle = listOfColors[index]
})


ctx.fillStyle = "white";


// Variables to track the current and previous mouse coordinates
var prevX, prevY;
var currX, currY;

// Flag to determine whether to start drawing
var isDrawing = false;




// Add an event listener to track mouse movements
canvas.addEventListener("pointermove", function (event) {
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
canvas.addEventListener("pointerdown", function (event) {
    //document.body.style.overflowY = "hidden";
    // Capture the starting mouse coordinates
    prevX = event.clientX - canvas.getBoundingClientRect().left;
    prevY = event.clientY - canvas.getBoundingClientRect().top;
    isDrawing = true;
});

// Add an event listener to stop drawing when the mouse button is released
canvas.addEventListener("pointerup", function () {
    isDrawing = false;
    //document.body.style.overflowY = "auto";
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
