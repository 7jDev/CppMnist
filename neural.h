#include "value.h"
#include <type_traits>
#ifndef NEURON_H
#define NEURON_H
enum activations{
	TANH,
	ELU
};
class neuron
{
	private:
		value_array weights; 
		value bias; 
		value_array weights_input; 
		value sum;
		value sumwbias; 
		value out; 	
	public:
	neuron();
	neuron(size_t input); 
	void output(value_array& input,activations act);
	value& return_output() ;
};
#endif 
#ifndef MLP_H
#define MLP_H
class mlp 
{
	private:
		std::vector<int> layer_neurons; 
		std::vector<std::vector<neuron>> layers;
	public:

	mlp();
	mlp(std::initializer_list<int> temp);
	
	

};
#endif 
