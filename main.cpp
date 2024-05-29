#include "queue.h"
#include "value.h"
int main(){
	value_array x(3);
	x.random_init();
	value_array y(3);
	y.random_init();
	value_array g(x*y);
	value finale(std::move(g.sum()));
	finale.calculate_gradients();
}
