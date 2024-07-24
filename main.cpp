#include "value.h"
#include <iostream>
#include "neural_net.h"
int main(){
	value_array h(3);
	h.random_init();
	layer x(3, 64, TANH, true);
	x.forward_layer(h);
}
