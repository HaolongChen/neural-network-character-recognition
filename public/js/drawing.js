// Drawing Canvas Functionality
class DrawingCanvas {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.setupCanvas();
        this.setupEventListeners();
    }

    setupCanvas() {
        this.ctx.lineWidth = 15;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.strokeStyle = 'black';
        this.clearCanvas();
    }

    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));

        // Touch events for mobile support
        this.canvas.addEventListener('touchstart', this.startDrawingTouch.bind(this));
        this.canvas.addEventListener('touchmove', this.drawTouch.bind(this));
        this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));
    }

    startDrawing(e) {
        this.isDrawing = true;
        this.ctx.beginPath();
        const { x, y } = this.getCoordinates(e);
        this.ctx.moveTo(x, y);
    }

    startDrawingTouch(e) {
        e.preventDefault();
        if (e.touches && e.touches[0]) {
            this.isDrawing = true;
            this.ctx.beginPath();
            const { x, y } = this.getTouchCoordinates(e);
            this.ctx.moveTo(x, y);
        }
    }

    draw(e) {
        if (!this.isDrawing) return;
        const { x, y } = this.getCoordinates(e);
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
    }

    drawTouch(e) {
        e.preventDefault();
        if (!this.isDrawing || !e.touches || !e.touches[0]) return;
        const { x, y } = this.getTouchCoordinates(e);
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
    }

    stopDrawing() {
        this.isDrawing = false;
    }

    getCoordinates(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    getTouchCoordinates(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.touches[0].clientX - rect.left,
            y: e.touches[0].clientY - rect.top
        };
    }

    clearCanvas() {
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    getImageData() {
        // Return image data for neural network processing
        return this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    }

    // Create a small thumbnail version of the current drawing
    createThumbnail(width = 28, height = 28) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Draw the current canvas content onto the thumbnail canvas
        tempCtx.drawImage(this.canvas, 0, 0, width, height);
        
        return tempCanvas;
    }

    // Get normalized pixel data for the neural network - IMPROVED VERSION
    getProcessedData() {
        // Create a 28x28 downsampled version (common for MNIST-like datasets)
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Fill with white background
        tempCtx.fillStyle = 'white';
        tempCtx.fillRect(0, 0, 28, 28);
        
        // Resize the image to 28x28 with proper scaling
        // Determine the aspect ratio of the drawing
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const bounds = this.getDrawingBounds(imageData);
        
        if (bounds.isEmpty) {
            // Return empty data if nothing is drawn
            return new Float32Array(28 * 28);
        }
        
        // Center the drawing with padding
        const padding = 2; // Add padding for better recognition
        const sourceX = Math.max(0, bounds.minX - padding);
        const sourceY = Math.max(0, bounds.minY - padding);
        const sourceWidth = Math.min(this.canvas.width - sourceX, bounds.maxX - bounds.minX + padding * 2);
        const sourceHeight = Math.min(this.canvas.height - sourceY, bounds.maxY - bounds.minY + padding * 2);
        
        // Calculate destination dimensions to maintain aspect ratio
        let destWidth, destHeight, destX, destY;
        
        if (sourceWidth > sourceHeight) {
            destWidth = 20; // Leave margin for 28x28
            destHeight = Math.round((sourceHeight * destWidth) / sourceWidth);
            destX = 4; // Center horizontally
            destY = Math.round((28 - destHeight) / 2);
        } else {
            destHeight = 20; // Leave margin for 28x28
            destWidth = Math.round((sourceWidth * destHeight) / sourceHeight);
            destY = 4; // Center vertically
            destX = Math.round((28 - destWidth) / 2);
        }
        
        // Draw the centered image
        tempCtx.drawImage(
            this.canvas, 
            sourceX, sourceY, sourceWidth, sourceHeight,
            destX, destY, destWidth, destHeight
        );
        
        // Get the image data
        const processedImageData = tempCtx.getImageData(0, 0, 28, 28);
        
        // Convert to grayscale and normalize to [0, 1]
        const data = new Float32Array(28 * 28);
        for (let i = 0; i < processedImageData.data.length; i += 4) {
            // Average the RGB values to get grayscale
            const avg = (processedImageData.data[i] + processedImageData.data[i + 1] + processedImageData.data[i + 2]) / 3;
            // Normalize to [0, 1] and invert (so 1 is dark and 0 is light, like MNIST)
            data[i / 4] = (255 - avg) / 255;
        }
        
        return data;
    }
    
    // Helper function to find the bounds of the drawing
    getDrawingBounds(imageData) {
        let minX = this.canvas.width;
        let minY = this.canvas.height;
        let maxX = 0;
        let maxY = 0;
        let isEmpty = true;
        
        // Find the boundaries of the drawing by looking for non-white pixels
        for (let y = 0; y < this.canvas.height; y++) {
            for (let x = 0; x < this.canvas.width; x++) {
                const i = (y * this.canvas.width + x) * 4;
                
                // Check if this pixel is not white (drawing)
                if (imageData.data[i] < 250 || imageData.data[i+1] < 250 || imageData.data[i+2] < 250) {
                    isEmpty = false;
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }
        
        return { minX, minY, maxX, maxY, isEmpty };
    }

    // Draw the given image data onto the canvas
    drawImageData(imageData, width = 28, height = 28) {
        // Resize canvas to match image dimensions if needed
        this.canvas.width = width;
        this.canvas.height = height;
        
        // Create a new ImageData object
        const drawImageData = new ImageData(width, height);
        
        // Copy pixel data
        for (let i = 0; i < imageData.length; i++) {
            const value = Math.round(imageData[i] * 255);
            const idx = i * 4;
            drawImageData.data[idx] = value; // R
            drawImageData.data[idx + 1] = value; // G
            drawImageData.data[idx + 2] = value; // B
            drawImageData.data[idx + 3] = 255; // A
        }
        
        // Draw the image data onto the canvas
        this.ctx.putImageData(drawImageData, 0, 0);
    }
}