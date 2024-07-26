#include "value.h"
#include <iostream>
#include "neural_net.h"
int main(){
	value_array h(784);
	h.random_init();
	layer x(784, 784, TANH, false);
	x.normal_forward_layer(h);
}
