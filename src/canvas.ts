
//
// Index file is just for setting up the canvas and providing sensible renderign APIs.
//

const canvas = document.getElementById("canvas");
const ctx    = canvas.getContext("2d");


// Config

const BLACK = "#212121";
const WHITE = "#fffefc";


// Functions

const setupCanvas = () => {
  canvas.style.position = "absolute";
  canvas.style.top = "0px";
  canvas.style.left = "0px";
  document.body.style.backgroundColor = BLACK;
  setCanvasSize();
  resetCanvas();
}

const setCanvasSize = () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  resetCanvas();
}

const resetCanvas = () => {
  ctx.fillStyle = WHITE;
}



// Listenres

window.addEventListener('resize', setCanvasSize);


// Init

setupCanvas();


// Simplified drawing API



