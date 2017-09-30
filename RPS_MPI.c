#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>

#include "RPS_MPI.h"

void initialize();
void initialize_petri();
void exchange_borders();
void iterate_CA();
void gather_petri();
void create_types();

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

// Each process is responsible for one part of the petri dish.
// Since we can't update the petri-dish in place each process actually
// gets two petri-dishes which they update in a lockstep fashion.
// dish A is updated by writing to dish B, then next step dish B updates dish A.
// (or you can just swap them inbetween iterations)
cell* local_petri_A;
cell* local_petri_B;

cell** local_petri_A_image;
cell** local_petri_B_image;

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


  // A super basic example sending some data:

  // cell* my_test_cell = malloc(10*sizeof(cell));
  // for(int ii = 0; ii < 10; ii++){
  //   my_test_cell[ii].strength = ii;
  //   my_test_cell[ii].color = rank;
  // }

  // if(rank == 0){
  //   cell* rec_buf = malloc(sizeof(cell)*10);
  //   for(int ii = 0; ii < size - 1; ii++){
  //     MPI_Recv(rec_buf, 10, mpi_cell_t, ii+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //     printf("receiving from rank %d: \n", ii+1);
  //     for(int jj = 0; jj < 10; jj++){
  //       printf("[%d, %d]  ", rec_buf[jj].color, rec_buf[jj].strength);
  //     }
  //     printf("\n");
  //   }
  // }
  // else{
  //   MPI_Send(my_test_cell, 10, mpi_cell_t, 0, 0, MPI_COMM_WORLD);
  // }


  gather_petri();

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
  MPI_Type_contiguous(p_local_petri_x_dim * p_local_petri_y_dim,
                      mpi_cell_t,
                      &local_petri_t);
  MPI_Type_commit(&local_petri_t);
  ////////////////////////////////
  ////////////////////////////////


  	//TODO: Create MPI types for border exchange

	
}


void initialize(){
  //TODO: assign the following to something more useful than 0
	/* Each dimension is increased by 2 to let there be a blank border surrounding the actual petri cells for which to insert information retrieved from border exchange */
  p_local_petri_x_dim = IMG_X / sqrt(size) + 2;
  p_local_petri_y_dim = IMG_Y / sqrt(size) + 2;

  // TODO: When allocating these buffers, keep in mind that you might need to allocate a little more
  // than just your piece of the petri.
  local_petri_A = malloc((p_local_petri_x_dim * p_local_petri_y_dim)*sizeof(cell)); // "Might need a little more"
  local_petri_B = malloc((p_local_petri_x_dim * p_local_petri_y_dim)*sizeof(cell)); // "Might need a little more"

  for (int i = 0; i < p_local_petri_x_dim*p_local_petri_y_dim; i++) {

		local_petri_A[i].strength = 0;
		local_petri_A[i].color = 0;

	}


  // TODO: Randomly perturb the local dish. Only perturb cells that belong to your process,
  // leave border pixels white.

	srand(rank);

	if (rank == 0) {

	memcpy(local_petri_B, local_petri_A, (p_local_petri_x_dim*p_local_petri_y_dim)*sizeof(cell)); 

	cell** local_petri_A_image = malloc(p_local_petri_x_dim*sizeof(cell*));
	for (int i = 0; i < p_local_petri_x_dim; i++) {
		local_petri_A_image[i] = &local_petri_A[(p_local_petri_x_dim * i)];
	}

	cell** local_petri_B_image = malloc(p_local_petri_x_dim*sizeof(cell*));
	for (int i = 0; i < p_local_petri_x_dim; i++) {
		local_petri_B_image[i] = &local_petri_B[(p_local_petri_x_dim * i)];
	}

	
	// "Randomly" seeding the local dish.
	for (int i = 0; i < p_local_petri_x_dim; i++) {

		int r = rand() % (p_local_petri_x_dim*p_local_petri_y_dim - 1);
		int rt = rand() % 4;
		
		local_petri_A[r].color = rt;
		local_petri_A[r].strength = 1;

	} 

//	printf("My dims are: %d %d\n", p_my_y_dim, p_my_x_dim);

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


/* 	if (p_my_y_dim == 0) {
		for (int i = 0; i < p_local_petri_y_dim; i++) {

			local_petri_A[i*p_local_petri_x_dim].color = 0;
			local_petri_A[i*p_local_petri_x_dim].strength = 0;	
		}
	} if (p_my_x_dim == 0) {
		for (int i = 0; i < p_local_petri_x_dim; i++) {

			local_petri_A[i].color = 0;
			local_petri_A[i].strength = 0;		
			
		}
	} if (p_my_y_dim == sqrt(size)-1) {
		for (int i = 0; i < p_local_petri_y_dim; i++) {

			local_petri_A[i*p_local_petri_y_dim + (p_local_petri_x_dim - 1)].color = 0;
			local_petri_A[i*p_local_petri_y_dim + (p_local_petri_x_dim - 1)].strength = 0;	
		}
	} if (p_my_x_dim == sqrt(size)-1) {
		for (int i = 0; i < p_local_petri_x_dim; i++) {

			local_petri_A[(p_local_petri_y_dim - 1)*p_local_petri_x_dim + i].color = 0;
			local_petri_A[(p_local_petri_y_dim - 1)*p_local_petri_x_dim + i].strength = 0;
		}

	}

*/	for (int i = 0; i < p_local_petri_y_dim * p_local_petri_x_dim; i++) {


		if (i%(p_local_petri_y_dim) == 0) {
			printf("%d: ", rank);
		}
		printf("(%d, %d)", local_petri_A[i].color, local_petri_A[i].strength);

		if (i%(p_local_petri_y_dim) == p_local_petri_y_dim-1) {
			printf("\n");
		}

	}





//	printf("(%d, %d)\n", local_petri_B_image[3][1].color, local_petri_B_image[1][3].strength);
}


//	printf("My rank is %d; coords are %d %d\n", rank, p_my_y_dim, p_my_x_dim); // coords: y, x
//  	printf("Rank %d: My neighbours are west %d, east: %d, north: %d, south: %d\n", rank, 	p_west, p_east, p_north, p_south); // To be removed
  
}


void exchange_borders(){
  //TODO: Exchange borders inbetween each step

	

}

void iterate_CA(){
  //TODO: Iterate the cellular automata one step
}

void gather_petri(){
  //TODO: Gather the final petri for process rank 0
}
