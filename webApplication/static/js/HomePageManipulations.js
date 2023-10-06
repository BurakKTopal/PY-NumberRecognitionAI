document.addEventListener("DOMContentLoaded", function () {
var TNN = document.getElementById("TPN");
var TNP = document.getElementById("TPP");

TPN.addEventListener('click', function(){

TPN.style.display = "none"
TPP.style.display = "block"
})

TPP.addEventListener('click', function(){
TPN.style.display = "block"
TPP.style.display = "none"
})
})