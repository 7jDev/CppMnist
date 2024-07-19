#include "queue.h"
#include "value.h"
#include "neural.h"
int main(){
mlp x(784, {64,32}, {ELU,ELU});
x.one_epoch();
}
