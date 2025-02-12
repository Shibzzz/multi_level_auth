document.addEventListener("DOMContentLoaded", () => {
    const puzzleGrid = document.getElementById("puzzle-grid");
    const imageSrc = "/path/to/user/image.jpg"; // Replace with the user's selected image
    const gridSize = 3; // 3x3 grid
    let shuffledOrder = []; // Store the shuffled order of pieces

    // Generate grid pieces
    function generateGrid() {
        const pieces = [];
        for (let i = 0; i < gridSize * gridSize; i++) {
            pieces.push(i);
        }
        shuffledOrder = pieces.sort(() => Math.random() - 0.5); // Shuffle pieces

        shuffledOrder.forEach((pos, index) => {
            const piece = document.createElement("div");
            piece.classList.add("puzzle-piece");
            piece.style.backgroundImage = `url(${imageSrc})`;
            piece.style.backgroundPosition = `${-(pos % gridSize) * 100}px ${-Math.floor(pos / gridSize) * 100}px`;
            piece.draggable = true;
            piece.dataset.position = pos;
            puzzleGrid.appendChild(piece);

            // Drag-and-Drop Event Listeners
            piece.addEventListener("dragstart", dragStart);
            piece.addEventListener("dragover", dragOver);
            piece.addEventListener("drop", drop);
        });
    }

    let draggedPiece = null;

    function dragStart(event) {
        draggedPiece = event.target;
    }

    function dragOver(event) {
        event.preventDefault();
    }

    function drop(event) {
        const targetPiece = event.target;

        // Swap the positions of draggedPiece and targetPiece
        if (draggedPiece && targetPiece) {
            const tempPosition = draggedPiece.dataset.position;
            draggedPiece.dataset.position = targetPiece.dataset.position;
            targetPiece.dataset.position = tempPosition;

            // Swap their visual order
            puzzleGrid.insertBefore(draggedPiece, targetPiece.nextSibling);
            puzzleGrid.insertBefore(targetPiece, draggedPiece.nextSibling);
        }
    }

    // Validate pattern on submission
    document.getElementById("submit-pattern").addEventListener("click", () => {
        const currentOrder = Array.from(puzzleGrid.children).map(piece => piece.dataset.position);
        console.log("Current Order:", currentOrder);
        // Send currentOrder to the server for validation
    });

    generateGrid();
});
