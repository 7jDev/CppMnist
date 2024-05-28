#include "queue.h"
#include "value.h"
int main(){
	value x(0.5);
	value y(0.5);
	value z(x+y);
	value last(z.tanh());
	last.calculate_gradients();
}
