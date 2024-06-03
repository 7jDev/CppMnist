#include "queue.h"
#include "value.h"
#include "neural.h"
int main(){
	value_array temp(1000);
	temp.random_init();
	layer x(2,1000, TANH);
       x.l_output(temp);	
}
