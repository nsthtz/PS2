#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

#include "RPS_MPI.h"

// Function declarations
void initialize();
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
int p_local_petri_x_dim_padded;
int p_local_petri_y_dim_padded;

int p_local_petri_x_dim;
int p_local_petri_y_dim;

MPI_Comm cart_comm;

// some datatypes, useful for sending data with somewhat less primitive semantics
MPI_Datatype border_row_t;  // TODO: Implement this
MPI_Datatype border_col_t;  // TODO: Implement this
MPI_Datatype local_petri_t; // Already implemented
MPI_Datatype mpi_cell_t;    // Already implemented
MPI_Datatype petri_selection_t; // Type for collecting the real cells within a padded petri.
MPI_Datatype petri_selection_global_t; // Type for inserting cells with the correct stride for a global petri.

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

// For the global petri when gathering
cell* global_petri;
cell** global_petri_image;

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

	// A counter to keep track of which petri to iterate on.
	petri_counter = 2;

	for (int i = 0; i < ITERATIONS; i++) {

		if (petri_counter % 2 == 0) {

			exchange_borders(local_petri_A);

			iterate_CA(local_petri_A_image, local_petri_B_image);

		} else {

			exchange_borders(local_petri_B);

			iterate_CA(local_petri_B_image, local_petri_A_image);
		}

		petri_counter++;

		/* Printing the current iteration number to have something to watch while waiting. I have no idea whether this affects the run-time significantly or not. Remove if it does. */
		if (rank == 0) { printf("%3d\n", petri_counter - 2); }		
		
	} 

	if (petri_counter % 2 == 0) {
		gather_petri(local_petri_B);
	} else {
		gather_petri(local_petri_A);
	}

	// Freeing all MPI types before MPI_Finalize(), since the terminal commented on it.
	
	MPI_Type_free(&mpi_cell_t);
	MPI_Type_free(&local_petri_t);	
	MPI_Type_free(&border_row_t);
	MPI_Type_free(&border_col_t);
	MPI_Type_free(&petri_selection_t);
	MPI_Type_free(&petri_selection_global_t);

	MPI_Finalize();

	if(rank==0){
	// TODO: Write the petri to an image

		// Same usage as in the serial version.
		make_bmp(global_petri_image, petri_counter-2);
	
	}

	// You should probably make sure to free your memory here
	// We will dock points for memory leaks, don't let your hard work go to waste!
	// free_stuff()

	// Freeing all allocated memory
	
	free(local_petri_A);
	free(local_petri_B);
	free(local_petri_A_image);
	free(local_petri_B_image);
	free(global_petri);
	free(global_petri_image);


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
	MPI_Type_contiguous(p_local_petri_x_dim_padded * p_local_petri_y_dim_padded, mpi_cell_t, 	&local_petri_t);
	MPI_Type_commit(&local_petri_t);
	

  	////////////////////////////////
  	////////////////////////////////


  	//TODO: Create MPI types for border exchange

	// A type for rows in border exchanges

	MPI_Type_contiguous(p_local_petri_y_dim_padded, mpi_cell_t, &border_row_t);
  	MPI_Type_commit(&border_row_t);

	// A type for columns in border exchanges

	MPI_Type_vector(p_local_petri_x_dim_padded, 1, p_local_petri_x_dim_padded, mpi_cell_t, &border_col_t);
	MPI_Type_commit(&border_col_t);

	// A type for selecting the pertinent parts of a padded array

	MPI_Type_vector(p_local_petri_x_dim, p_local_petri_y_dim, p_local_petri_x_dim_padded, mpi_cell_t, &petri_selection_t);
	MPI_Type_commit(&petri_selection_t);

	// A type for inserting the pertinent parts of an array into the global petri
	
	MPI_Type_vector(p_local_petri_x_dim, p_local_petri_y_dim, IMG_X, mpi_cell_t, &petri_selection_global_t);
	MPI_Type_commit(&petri_selection_global_t);

}

void initialize(){

	//TODO: assign the following to something more useful than 0
	
	/* Each dimension is increased by 2 to let there be a blank border surrounding the actual petri cells for which to insert information retrieved from border exchange.
	NB: The following could have been inserted into the initialize_petri() function declared earlier, but seeing as the sample code already had begun doing it here, I just finished the last couple of lines in place */

	p_local_petri_x_dim = IMG_X / p_x_dims;
	p_local_petri_y_dim = IMG_Y / p_y_dims;

	p_local_petri_x_dim_padded = p_local_petri_x_dim + 2*BORDER_SIZE;
	p_local_petri_y_dim_padded = p_local_petri_y_dim + 2*BORDER_SIZE;

	// TODO: When allocating these buffers, keep in mind that you might need to allocate a little more
	// than just your piece of the petri.

	// Enough space is allocated by using the padded dim values when working on local petris.
	
	local_petri_A = calloc((p_local_petri_x_dim_padded * p_local_petri_y_dim_padded), sizeof(cell));
	local_petri_B = calloc((p_local_petri_x_dim_padded * p_local_petri_y_dim_padded), sizeof(cell));

	/* Images similar to the serial version are used at some points in the program, especially when interfacing when pre-made functions from CA. Otherwise it operates directly on 1D-arrays. */

	local_petri_A_image = malloc(p_local_petri_x_dim_padded*sizeof(cell*));
	for (int i = 0; i < p_local_petri_x_dim_padded; i++) {
		local_petri_A_image[i] = &local_petri_A[(p_local_petri_x_dim_padded * i)];
	}

	local_petri_B_image = malloc(p_local_petri_x_dim_padded*sizeof(cell*));
	for (int i = 0; i < p_local_petri_x_dim_padded; i++) {
		local_petri_B_image[i] = &local_petri_B[(p_local_petri_x_dim_padded * i)];
	} 



	// TODO: Randomly perturb the local dish. Only perturb cells that belong to your process,
	// leave border pixels white.


	/* "Randomly" seeding the local dish, with more or less the same frequency as the serial program. */
	
	srand(rank);

	for (int i = 0; i < IMG_X/size; i++) {

		int rx = rand() % (p_local_petri_x_dim_padded - 3);
		int ry = rand() % (p_local_petri_y_dim_padded - 3);
		int rt = rand() % 4;
		
		local_petri_A_image[1+rx][1+ry].color = rt;
		local_petri_A_image[1+rx][1+ry].strength = 1;

	} 

	/* Setting all borders to white */

	for (int i = 0; i < p_local_petri_x_dim_padded; i++) {
		
		local_petri_A_image[i][0].color = 0;
		local_petri_A_image[i][0].strength = 0;

		local_petri_A_image[i][p_local_petri_x_dim_padded-1].color = 0;
		local_petri_A_image[i][p_local_petri_x_dim_padded-1].strength = 0;

		local_petri_A_image[0][i].color = 0;
		local_petri_A_image[0][i].strength = 0;

		local_petri_A_image[p_local_petri_y_dim_padded-1][i].color = 0;
		local_petri_A_image[p_local_petri_y_dim_padded-1][i].strength = 0;

	}	

}

void exchange_borders(cell* petri){

  	//TODO: Exchange borders inbetween each step

	/* Exchange north if applicable using the border_row_t MPI_type, using relative to dimension indexes. I also use a buffer for each row exchanged that is inserted into the petri at the appropriate position. */


		MPI_Sendrecv(&petri[p_local_petri_x_dim_padded], 1, border_row_t, p_north, 0, &petri[0], 1, border_row_t, p_north, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		int send_index = (p_local_petri_x_dim_padded - 2) * p_local_petri_x_dim_padded;
		int recv_index = (p_local_petri_x_dim_padded - 1) * p_local_petri_x_dim_padded;

		MPI_Sendrecv(&petri[send_index], 1, border_row_t, p_south, 0, &petri[recv_index], 1, border_row_t, p_south, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


	/* Exchange east/west if applicable using the border_col_t MPI_type, using relative to dimension indexes. Here I do not use a buffer, as the resulting vector spans the original length of the petri, so I rather just overwrite the appropriate values in the petri directly. */
 
		send_index = p_local_petri_y_dim_padded - 2;
		recv_index = p_local_petri_y_dim_padded - 1;
 
		MPI_Sendrecv(&petri[send_index], 1, border_col_t, p_east, 0, &petri[recv_index], 1, border_col_t, p_east, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		MPI_Sendrecv(&petri[1], 1, border_col_t, p_west, 0, petri, 1, border_col_t, p_west, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


	/* NB: The resulting exchange DOES also contain diagonal values at the corners of the petri, as the exchange is propagated (1) from north to south, then from east to west using the transferred border values of (1) already in place.
Support for border size adjustment is not implemented */

}

void iterate_CA(cell** old_image, cell** next_image){
//TODO: Iterate the cellular automata one step

	// Uses CA functions with only minor changes.

	iterate_image(old_image, next_image);
	
}

void gather_petri(cell* newest_petri){
//TODO: Gather the final petri for process rank 0

	// root process holds the global petri, inserting its own local petri values first.

	if (rank == 0) {
		
		global_petri = calloc(IMG_X*IMG_Y, sizeof(cell));	
		
		for (int y = 1; y < p_local_petri_y_dim_padded - 1; y++) {		

			memcpy(&global_petri[IMG_Y*(y-1)], &newest_petri[(p_local_petri_y_dim_padded*y)+1], (p_local_petri_x_dim)*sizeof(cell));
		}
	
		/* It then receives in turn all local petri's from other ranks, using their ranks and the known dimensions of the cart_comm to calculate the correct placement in the global petri. NB: root process converts the petri_selection_t vector to a petri_selection_global_t that has the correct stride value for the larger global petri. */

		for (int r = 1; r < size; r++) {

			// index-by-rank formula
			int i = (r%p_x_dims * (p_local_petri_x_dim)) + (IMG_Y / p_y_dims) * ((r)/p_y_dims)*IMG_X;

			MPI_Recv(&global_petri[i], 1, petri_selection_global_t, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		}

		// A 2D image is created to be used when making the .bmp
		global_petri_image = malloc(IMG_Y*sizeof(cell**));
	
		for (int i = 0; i < IMG_Y; i++) {

			global_petri_image[i] = &global_petri[i*IMG_Y];

		}


	} else {

		/* All other processes just send the relevant piece of their local petri's, using the petri_selection_t vector 	specified previously. */
		MPI_Send(&newest_petri[p_local_petri_x_dim_padded + 1], 1, petri_selection_t, 0, 0, MPI_COMM_WORLD);
	}
	
}

/* CA functions mostly just copied from CA.c as with the 2d array implementation, each local petri should behave the same as in the serial version. The only changes are commented. */

cell pick_neighbor(int x, int y, cell** image);

bool beats(cell me, cell other){
	return
	(((me.color == SCISSOR) && (other.color == PAPER)) ||
	((me.color == PAPER) && (other.color == ROCK))    ||
	((me.color == ROCK) && (other.color == SCISSOR))  ||
	(me.color == other.color));
}

cell next_cell(int x, int y, cell** image){
	
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
		return image[x + c_x - 1][y + c_y - 1];
	}

void iterate_image(cell** old_image, cell** next_image){
	
	// Using local dims instead of global dims.
	for(int xx = 1; xx < p_local_petri_x_dim_padded - 1; xx++){
		for(int yy = 1; yy < p_local_petri_y_dim_padded - 1; yy++){

			next_image[xx][yy] = next_cell(xx, yy, old_image);

		}
	} 

}

	/* Print functions used for debugging below. 
	NB: some might not be updated, the functions were sometimes altered to suit a specific need, and no revisions were made */

void print_petri(cell* petri) {

	for (int i = 0; i < p_local_petri_y_dim_padded * p_local_petri_x_dim_padded; i++) {


		if (i%(p_local_petri_y_dim_padded) == 0) {
			printf("%d: ", rank);
		}
		printf("(%d, %d)", petri[i].color, petri[i].strength);

		if (i%(p_local_petri_y_dim_padded) == p_local_petri_y_dim-1) {
			printf("\n");
		}

	}
	printf("\n");

}

void print_global_petri(cell* global_petri) {

	for (int i = 0; i < IMG_X * IMG_Y; i++) {


		printf("(%2d, %2d)", global_petri[i].color, global_petri[i].strength);

		if (i%(IMG_Y) == IMG_Y-1) {
			printf("\n");
		}

	}
	printf("\n");

}

void print_image(cell** image) {

	for (int y = 0; y < IMG_Y; y++) {
		printf("%d: ", rank);
		for (int x = 0; x < IMG_X; x++) {
			
			printf("(%d, %d) ", image[y][x].color, image[y][x].strength);
		}
		printf("\n");
	}
	printf("\n");
}


void print_border(cell* border) {

	for (int i = 0; i < p_local_petri_y_dim_padded; i++) {


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
