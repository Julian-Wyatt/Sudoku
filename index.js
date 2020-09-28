class sudoku{

	constructor(){

		this.grid = []
		this.tempGrid = []
		
		
	}


    // # https://www.youtube.com/watch?v=G_UYXzGuqvM
	solve(grid){
		// this.grid[0][0] = 8
		// // this.solve()
		// console.log(this.grid)

		// return
		for (let i=0;i<=8;i++){
			for (let j=0;j<=8;j++){
				if (grid[i][j]==0){
					for (let k=1;k<=9;k++){
						if (this.possible(i,j,k)){
							grid[i][j] = k
							this.solve(grid)
							grid[i][j] = 0
							
						}
					}
					return
				}
			}
		}
		this.grid = JSON.parse(JSON.stringify(grid))
	}
	possible(x,y,n){
		`Is n possible in position x,y`
		for (let i=0;i<9;i++){
			if (this.grid[x][i] == n){
				return false
			}
			if (this.grid [i][y]==n){
				return false
			}
		}

		let x0 = (Math.floor(x/3))*3
		let y0 = (Math.floor(y/3))*3

		for (let i=0;i<=2;i++){
			for (let j=0;j<=2;j++){
				if (this.grid[x0+i][y0+j]==n){
					return false
				}
			}
		}

        return true
	}


}

var sudokuGrid;
function main() {
	// document.getElementById("blah").innerHTML
	sudokuGrid = new sudoku()
	updateGridObject(sudokuGrid)
	
}

function updateGridObject(grid){
	grid.grid = []
	for (let i=0;i<9;i++){
		let row = document.getElementById("row"+(i+1)).children
		grid.grid[i] = [] 
		for (let j=0;j<9;j++){
			// console.log(row.item(j).innerHTML)
			if (row.item(j).innerHTML == ""){
				grid.grid[i][j] = 0
			}
			else{
				grid.grid[i][j] = parseInt(row.item(j).innerHTML)

			}
		}
		// grid.grid.push(rowArray)

	}
	console.log("updating object")
	console.log(grid.grid)
}
function updateGridHTML(grid){
	for (let i=0;i<9;i++){
		let row = document.getElementById("row"+(i+1)).children

		for (let j=0;j<9;j++){

			// console.log(row.item(j).innerHTML)
			if (grid.grid[i][j].toString() == "0"){
				row.item(j).innerHTML = ""
			}
			else{
				row.item(j).innerHTML = grid.grid[i][j].toString()

			}
		}
		// grid.grid.push(rowArray)

	}
	console.log("updating HTML")
	console.log(grid.grid)
}

function clearGridHTML(){
	for (let i=0;i<9;i++){
		let row = document.getElementById("row"+(i+1)).children

		for (let j=0;j<9;j++){
			row.item(j).innerHTML = ""
			
		}
		// grid.grid.push(rowArray)

	}
	updateGridObject(sudokuGrid)
}

function onOpenCvReady() {
	document.getElementById('status').innerHTML = 'OpenCV.js is ready.';
  }

let keypressed = {}
document.addEventListener("DOMContentLoaded",function(){

	

	// Key events and animations
	document.onkeydown = function (ev) {

		// https://stackoverflow.com/questions/35394937/keyboardevent-keycode-deprecated-what-does-this-mean-in-practice
		var code;
		if (ev.key !== undefined) {
			code = ev.key;
			if (code >=0 && code <=9){
				updateGridObject(sudokuGrid)
			}
		} 
		else if (ev.keyIdentifier !== undefined) {
			code = ev.keyIdentifier;
			if (code >=0 && code <=9){
				updateGridObject(sudokuGrid)
			}
		} 
		else if (ev.keyCode !== undefined) {
			code = ev.keyCode;

			// 48 is 0
			// 49 is 1 etc
			if (code >=48 && code <=57){
				updateGridObject(sudokuGrid)
			}
		}


		// for events on update/ multi key presses
		keypressed[code] = true;

	}

	let solveBtn = document.getElementById("solveButton")
	function solve (){
		sudokuGrid.solve(sudokuGrid.grid)
		console.log(sudokuGrid)
		updateGridHTML(sudokuGrid)
	}
	solveBtn.addEventListener("click",solve)

	let clearBtn = document.getElementById("clearButton")

	clearBtn.addEventListener("click",clearGridHTML)


	// TODO: Add step-by-step animation of how backtracking works - with skip one step, skip 10 steps or skip 100 steps
	// TODO: add computer vision - OpenCV aspect
	// TODO: Add Tensorflow model
	let imgElement = document.getElementById('imageSrc');
	let inputElement = document.getElementById('fileInput');
	inputElement.addEventListener('change', (e) => {
	  imgElement.src = URL.createObjectURL(e.target.files[0]);
	}, false);
	imgElement.onload = function() {
	  let mat = cv.imread(imgElement);
		let dsize = new cv.Size(0, 0);
		// You can try more different parameters
		cv.resize(mat, mat, dsize, 0.5, 0.5);
	  cv.imshow('canvasOutput', mat);
	console.log(mat)

	  mat.delete();

	};


	/**
	 * @function
	 * clear button - when clicked, calls clear function - also changes text of current blend mode back to blend
	 */
	// let clearBtn = document.getElementById("clearButton");
	// function callClear () {
	// 	attractor.clearButtonFunc();
	// 	document.getElementById("blendOut").textContent = "Blend";
	// }
	// clearBtn.addEventListener("click",callClear)
});
