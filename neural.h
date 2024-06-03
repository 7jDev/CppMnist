#include "value.h"
#include <type_traits>
#ifndef NEURON_H
#define NEURON_H
enum activations{
	TANH,
	ELU, 
	NONE
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
		activations neuron_activation;	
	public:
	neuron();
	neuron(size_t input,activations act); 
	void n_output(value_array& input);
	value& return_output() ;
};
#endif
#ifndef LAYER_H
#define LAYER_H
class layer{
	private:
		std::vector<neuron> m_neurons; 
		size_t m_size; 
		size_t m_input;
		activations m_func; 
		value_array output; 
	public:
		layer();
		layer(size_t amount, size_t input, activations act);
		void l_output(value_array& input); 
};
#endif
#ifndef MLP_H
#define MLP_H
class mlp 
{
	private:
		std::vector<int> layer_neurons; 
		std::vector<layer>layers;
		std::vector<activations> functions; 
		int input; 	
	public:

	mlp();
	mlp(int input_size,  std::initializer_list<int> temp, std::initializer_list<activations> activation_func);
};

#endif 
