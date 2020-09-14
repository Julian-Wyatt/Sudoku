class sudoku{

	constructor(table){
		this.elements = table
		console.log(this.elements)

	}


}

function main() {

	let instance = new sudoku(document.getElementById("grid"))
}

document.addEventListener("DOMContentLoaded",function(){

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

function solve(){
	instance = new sudoku(document.getElementById("grid"))
}