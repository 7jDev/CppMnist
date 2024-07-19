#include "value.h"
#include "neural_net.h"
int main(){
	value_array h(3);
	h.random_init();
	layer x(3, 2, TANH);
	x.normal_forward_layer(h);
}
