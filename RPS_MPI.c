#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

#include "RPS_MPI.h"

void initialize();
void initialize_petri();
void exchange_borders(cell* petri);
void iterate_CA(cell** old_image, cell** next_image);
void gather_petri(cell* newest_petri);
void create_types();
// For debugging
void print_petri(cell* petri);
void print_image(cell** petri_image);
void print_border(cell* border);
void print_global_petri(cell* global_petri);
void print_array(int* array);

int rank;
int size;

// I denote mpi process specific values with hungarian notation, adding a p

// The dimensions of the processor grid. Same for every process
int p_x_dims;
int p_y_dims;

// The location of a process in the process grid. Unique for every process
int p_my_x_dim;
int p_my_y_dim;

int p_north, p_south, p_east, p_west;

// The dimensions for the process local petri
int p_local_petri_x_dim;
int p_local_petri_y_dim;

MPI_Comm cart_comm;

// some datatypes, useful for sending data with somewhat less primitive semantics
MPI_Datatype border_row_t;  // TODO: Implement this
MPI_Datatype border_col_t;  // TODO: Implement this
MPI_Datatype local_petri_t; // Already implemented
MPI_Datatype mpi_cell_t;    // Already implemented
MPI_Datatype petri_selection_t; // Type for collecting the real cells within a petri.

// Each process is responsible for one part of the petri dish.
// Since we can't update the petri-dish in place each process actually
// gets two petri-dishes which they update in a lockstep fashion.
// dish A is updated by writing to dish B, then next step dish B updates dish A.
// (or you can just swap them inbetween iterations)
cell* local_petri_A;
cell* local_petri_B;

// Declare 2D array to be used similarly to the serial code.
cell** local_petri_A_image;
cell** local_petri_B_image;

cell* global_petri;

// Declare a global flag for keeping track of which petri to iterate on.
int petri_counter;

int main(int argc, char** argv){

	srand(1234);

	// Ask MPI what size (number of processors) and rank (which process we are)
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	////////////////////////////////
	// Create cartesian communicator
	int dims[2];
	dims[0] = p_x_dims;
	dims[1] = p_y_dims;


	int periods[2]; // we set these to 0 because we are not interested in wrap-around
	periods[0] = 0;
	periods[1] = 0;

	int coords[2];
	coords[0] = p_my_x_dim;
	coords[1] = p_my_y_dim;

	MPI_Dims_create(size, 2, dims);
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
	MPI_Cart_coords(cart_comm, rank, 2, coords);

	MPI_Cart_shift(cart_comm, 0, 1, &p_north, &p_south);
	MPI_Cart_shift(cart_comm, 1, 1, &p_west, &p_east);

	p_x_dims = dims[0];
	p_y_dims = dims[1];

	p_my_x_dim = coords[0];
	p_my_y_dim = coords[1];
	////////////////////////////////
	////////////////////////////////

	initialize();


	create_types();
	petri_counter = 2;

	for (int i; i < ITERATIONS; i++) {

		if (petri_counter % 2 == 0) {
			exchange_borders(local_petri_A);

			iterate_CA(local_petri_A_image, local_petri_B_image);
		} else {

			exchange_borders(local_petri_B);

			iterate_CA(local_petri_B_image, local_petri_A_image);
		}

		petri_counter++;		
		
	}

	if (petri_counter % 2 == 0) {
		gather_petri(local_petri_B);
	} else {
		gather_petri(local_petri_A);
	}

	MPI_Finalize();

	if(rank==0){
	// TODO: Write the petri to an image
	}

	// You should probably make sure to free your memory here
	// We will dock points for memory leaks, don't let your hard work go to waste!
	// free_stuff()

	exit(0);
}


void create_types(){

	////////////////////////////////
	////////////////////////////////
	// cell type
	const int    nitems=2;
	int          blocklengths[2] = {1,1};
	MPI_Datatype types[2] = {MPI_INT, MPI_INT};
	MPI_Aint     offsets[2];

	offsets[0] = offsetof(cell, color);
	offsets[1] = offsetof(cell, strength);

	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_cell_t);
	MPI_Type_commit(&mpi_cell_t);
  	////////////////////////////////
  	////////////////////////////////



  	////////////////////////////////
  	////////////////////////////////
  	// A message for a local petri-dish
	MPI_Type_contiguous(p_local_petri_x_dim * p_local_petri_y_dim, mpi_cell_t, 	&local_petri_t);
	MPI_Type_commit(&local_petri_t);
	

  	////////////////////////////////
  	////////////////////////////////


  	//TODO: Create MPI types for border exchange

	MPI_Type_contiguous(p_local_petri_y_dim, mpi_cell_t, &border_row_t);
  	MPI_Type_commit(&border_row_t);

	// (count, blocklength, stride, oldtype)

	MPI_Type_vector(p_local_petri_x_dim, 1, p_local_petri_x_dim, mpi_cell_t, &border_col_t);
	MPI_Type_commit(&border_col_t);

	MPI_Type_vector(p_local_petri_x_dim - 2, p_local_petri_y_dim - 2, p_local_petri_x_dim, mpi_cell_t, &petri_selection_t);
	MPI_Type_commit(&petri_selection_t);
	
}


void initialize(){
	//TODO: assign the following to something more useful than 0
	/* Each dimension is increased by 2 to let there be a blank border surrounding the actual petri cells for which to insert information retrieved from border exchange */

	p_local_petri_x_dim = IMG_X / sqrt(size) + 2;
	p_local_petri_y_dim = IMG_Y / sqrt(size) + 2;

	// TODO: When allocating these buffers, keep in mind that you might need to allocate a little more
	// than just your piece of the petri.
	
	local_petri_A = calloc((p_local_petri_x_dim * p_local_petri_y_dim), sizeof(cell));
	local_petri_B = calloc((p_local_petri_x_dim * p_local_petri_y_dim), sizeof(cell));

	local_petri_A_image = malloc(p_local_petri_x_dim*sizeof(cell*));
	for (int i = 0; i < p_local_petri_x_dim; i++) {
		local_petri_A_image[i] = &local_petri_A[(p_local_petri_x_dim * i)];
	}

	local_petri_B_image = malloc(p_local_petri_x_dim*sizeof(cell*));
	for (int i = 0; i < p_local_petri_x_dim; i++) {
		local_petri_B_image[i] = &local_petri_B[(p_local_petri_x_dim * i)];
	} 



  // TODO: Randomly perturb the local dish. Only perturb cells that belong to your process,
  // leave border pixels white.

	srand(rank);

	// "Randomly" seeding the local dish.
	for (int i = 0; i < p_local_petri_x_dim*2; i++) {

		int rx = rand() % (p_local_petri_x_dim - 3);
		int ry = rand() % (p_local_petri_y_dim - 3);
		int rt = rand() % 4;
		
		local_petri_A_image[1+rx][1+ry].color = rt;
		local_petri_A_image[1+rx][1+ry].strength = 1;

	} 


	// Setting all borders to white

	for (int i = 0; i < p_local_petri_x_dim; i++) {
		
		local_petri_A_image[i][0].color = 0;
		local_petri_A_image[i][0].strength = 0;

		local_petri_A_image[i][p_local_petri_x_dim-1].color = 0;
		local_petri_A_image[i][p_local_petri_x_dim-1].strength = 0;

		local_petri_A_image[0][i].color = 0;
		local_petri_A_image[0][i].strength = 0;

		local_petri_A_image[p_local_petri_y_dim-1][i].color = 0;
		local_petri_A_image[p_local_petri_y_dim-1][i].strength = 0;

	}	

}

void exchange_borders(cell* petri){

  	//TODO: Exchange borders inbetween each step

	// Exchange north if applicable using the border_row_t MPI_type, using relative to dimension indexes. I also use a buffer for each row exchanged that is inserted into the petri at the appropriate position.

	if (p_north != -1) {

//		cell* n_border_row = malloc((p_local_petri_y_dim)*sizeof(cell));

		MPI_Sendrecv(&petri[p_local_petri_x_dim], 1, border_row_t, p_north, 0, &petri[0], 1, border_row_t, p_north, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		

//		memcpy(&local_petri_A[0], n_border_row, p_local_petri_y_dim*sizeof(cell));

//		free(n_border_row);

	} if (p_south != -1) {

//		cell* s_border_row = malloc((p_local_petri_y_dim)*sizeof(cell));

		MPI_Sendrecv(&petri[(p_local_petri_x_dim - 2) * p_local_petri_x_dim], 1, border_row_t, p_south, 0, &petri[(p_local_petri_x_dim - 1) * p_local_petri_x_dim], 1, border_row_t, p_south, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


//		memcpy(&local_petri_A[(p_local_petri_x_dim - 1) * p_local_petri_x_dim], s_border_row, p_local_petri_y_dim*sizeof(cell));

//		free(s_border_row);

	}

	
	// Exchange east/west if applicable using the border_col_t MPI_type, using relative to dimension indexes. Here I do not use a buffer, as the resulting vector spans the original length of the petri, so I rather just overwrite the appropriate values in the petri directly.

	if (p_east != -1) {
		
		MPI_Sendrecv(&petri[p_local_petri_y_dim - 2], 1, border_col_t, p_east, 0, &petri[p_local_petri_y_dim-1], 1, border_col_t, p_east, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
	} if (p_west != -1) {

		MPI_Sendrecv(&petri[1], 1, border_col_t, p_west, 0, petri, 1, border_col_t, p_west, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	} 

	// NB: The resulting exchange DOES also contain diagonal values at the eastern corners of the petri, as the exchange is propagated (1) from north to south, then from east to west using the transferred border values of (1) already in place. These values will not be used when iterating.

/*	if (rank == 6 || rank == 2) {

		print_petri(petri);
		printf("\n");

	} */

}

void iterate_CA(cell** old_image, cell** next_image){
  //TODO: Iterate the cellular automata one step

	iterate_image(old_image,  next_image);
	
}

void gather_petri(cell* newest_petri){
  //TODO: Gather the final petri for process rank 0

//	if (rank == 0 ) {
		global_petri = calloc(IMG_X*IMG_Y, sizeof(cell));
/*		print_petri(newest_petri);
		for (int y = 1; y < p_local_petri_y_dim - 1; y++) {
			memcpy(&global_petri[IMG_Y*(y-1)], &newest_petri[(p_local_petri_y_dim*y)+1], (p_local_petri_x_dim-2)*sizeof(cell));
		}
*/		
		int* recvcounts = malloc(size*sizeof(int));
		for (int i = 0; i < size; i++) {
			recvcounts[i] = (p_local_petri_x_dim-2)*(p_local_petri_y_dim-2);
		}

		int* displs = malloc(size*sizeof(int));
		int i = 0;
		for (int y = 0; y < sqrt(size); y++) {
			for (int x = 0; x < sqrt(size); x++) {
				displs[i] = y*(IMG_Y*2) + x*(p_local_petri_x_dim-2);
//				printf("rank %d: %d\n", i, displs[i]);
				i++;
			}	
		}
//		print_array(recvcounts);
//		print_array(displs);
		
//		print_global_petri(global_petri);

		// (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, ierror)

		MPI_Gatherv(&newest_petri[p_local_petri_y_dim+1], (p_local_petri_x_dim-2)*(p_local_petri_y_dim-2), petri_selection_t, &global_petri, recvcounts, displs, petri_selection_t, 0, MPI_COMM_WORLD);
//		print_global_petri(global_petri);
	} else {
		int* recvcounts = malloc(size*sizeof(int));
		for (int i = 0; i < size; i++) {
			recvcounts[i] = (p_local_petri_x_dim-2)*(p_local_petri_y_dim-2);
		}

		int* displs = malloc(size*sizeof(int));
		int i = 0;
		for (int y = 0; y < sqrt(size); y++) {
			for (int x = 0; x < sqrt(size); x++) {
				displs[i] = y*(IMG_Y*2) + x*(p_local_petri_x_dim-2);
//				printf("rank %d: %d\n", i, displs[i]);
				i++;
			}	
		}

//		print_array(recvcounts);
//		print_array(displs);

		MPI_Gatherv(&newest_petri[p_local_petri_y_dim+1], (p_local_petri_x_dim-2)*(p_local_petri_y_dim-2), petri_selection_t, 0, recvcounts, displs, petri_selection_t, 0, MPI_COMM_WORLD);
	
	}

	if (rank == 0) {
		print_global_petri(global_petri);
	}

}

// CA functions mostly just copied from CA.c as with the 2d array implementation, each local petri should behave the same as in the serial version. The only changes are commented.

cell pick_neighbor(int x, int y, cell** image);

cell** alloc_img(cell* buffer, int index) {
	cell** image = malloc(IMG_X*sizeof(cell*));

	for(int ii = 0; ii < IMG_X; ii++){
		image[ii] = &buffer[(IMG_X*IMG_Y*index) + IMG_X*ii];
		}

	return image;
}

void free_img(cell** image){
	for (int ii = 0; ii < IMG_X; ii++){
		free(image[ii]);
	}
	free(image);
}


bool beats(cell me, cell other){
	return
	(((me.color == SCISSOR) && (other.color == PAPER)) ||
	((me.color == PAPER) && (other.color == ROCK))    ||
	((me.color == ROCK) && (other.color == SCISSOR))  ||
	(me.color == other.color));
}

cell next_cell(int x, int y, cell** image){
	
	// printf("Cell (%d, %d) picks cell ", x, y);
	cell neighbor_cell = pick_neighbor(x, y, image);
	cell my_cell = image[x][y];
	
	if(neighbor_cell.color == WHITE){
		return my_cell;
	}
	cell next_cell = my_cell;

	if(my_cell.color == WHITE){
		next_cell.strength = 1;
		next_cell.color = neighbor_cell.color;
		return next_cell;
	}
	else {
		if(beats(my_cell, neighbor_cell)) {
			next_cell.strength++;
		}
		else {
			next_cell.strength--;
		}
	}

	if(next_cell.strength == 0){
		next_cell.color = neighbor_cell.color;
		next_cell.strength = 1;
	}

	if(next_cell.strength > 4){
		next_cell.strength = 4;
	}

	return next_cell;
	}


cell pick_neighbor(int x, int y, cell** image){
	int chosen = rand() % 8;

	if(chosen >= 4){ chosen++; } // a cell cant choose itself
		int c_x = chosen % 3;
		int c_y = chosen / 3;
		// printf("(%d, %d)\n", x + c_x - 1, y + c_y - 1);
		return image[x + c_x - 1][y + c_y - 1];
	}

void iterate_image(cell** old_image, cell** next_image){
	
	// Using local dims instead of global dims.
	for(int xx = 1; xx < p_local_petri_x_dim - 1; xx++){
		for(int yy = 1; yy < p_local_petri_y_dim - 1; yy++){
			// printf("%d %d\n: ", xx, yy);
			next_image[xx][yy] = next_cell(xx, yy, old_image);
		}
	} 

}

	// Debug functions below //

void print_petri(cell* petri) {

	for (int i = 0; i < p_local_petri_y_dim * p_local_petri_x_dim; i++) {


		if (i%(p_local_petri_y_dim) == 0) {
			printf("%d: ", rank);
		}
		printf("(%d, %d)", petri[i].color, petri[i].strength);

		if (i%(p_local_petri_y_dim) == p_local_petri_y_dim-1) {
			printf("\n");
		}

	}
	printf("\n");

}

void print_global_petri(cell* global_petri) {

	for (int i = 0; i < IMG_X * IMG_Y; i++) {


		printf("(%d, %d)", global_petri[i].color, global_petri[i].strength);

		if (i%(IMG_Y) == IMG_Y-1) {
			printf("\n");
		}

	}
	printf("\n");

}

void print_image(cell ** image) {

	for (int y = 0; y < p_local_petri_y_dim; y++) {
		printf("%d: ", rank);
		for (int x = 0; x < p_local_petri_x_dim; x++) {
			
			printf("(%d, %d) ", image[y][x].color, image[y][x].strength);
		}
		printf("\n");
	}
	printf("\n");
}


void print_border(cell* border) {

	for (int i = 0; i < p_local_petri_y_dim; i++) {


		if (i%(p_local_petri_y_dim) == 0) {
			printf("%d: ", rank);
		}
		printf("(%d, %d)", border[i].color, border[i].strength);

		if (i%(p_local_petri_y_dim) == p_local_petri_y_dim-1) {
			printf("\n");
		}

	}
	
}

void print_array(int* array) {

	for (int i = 0; i < size; i++) {
		
		printf("rank %d: %d\n", i, array[i]);

	}
	printf("\n");

}
