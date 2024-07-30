#include "neural_net.h"
#include <iostream>
#include<chrono>
ThreadPool layer::threads{};
neuron::neuron(){}
neuron::neuron(size_t input_size, activation func): weights(input_size),function(func)
{
	bias.random_init(); 
	weights.random_init(); 
	bias.requires_grad();
	weights.requires_grad();
}
void neuron::set_input(value_array& in)
{
	input = std::move(in.return_copy()); 
}
void neuron::forward()
{
	weights_input = std::move(input*weights);
	sum = std::move(weights_input.sum());
	sumwbias = std::move(sum + bias);
	switch(function){
	case TANH: 
		final = std::move(sumwbias.tanh());
		return; 
	case ELU: 
		final = std::move(sumwbias.elu());
		return; 
	case NONE: 
		break;
	}
}
value& neuron::neuron_output()
{
if (function == NONE)
	return sumwbias; 
return final;
}
layer::layer(){}

layer::layer(size_t input_size, size_t amount_of_neurons, activation func, bool threadable):
	function(func),
	final(amount_of_neurons),
	m_amount_of_neurons(amount_of_neurons)
{	
	if(threadable){
	size_t split = split_up();
	for(size_t i = 0; i< amount_of_neurons;i = i+split){ 
		std::vector<neuron> temp; 
		for(size_t j=0; j< split; ++j)
			temp.emplace_back(neuron(input_size, func));
		m_neurons_fast.push_back(std::move(temp));

	}
	assert(m_neurons_fast.size() == std::thread::hardware_concurrency()); 
	}
	else{
	m_neurons.reserve(amount_of_neurons); 
	for(size_t i=0; i< amount_of_neurons; i++)
		m_neurons.emplace_back(neuron(input_size,func)); 
	}
	
}
size_t layer::split_up()
{
	static size_t number_of_threads; 
	if(!threads.is_started())
		 number_of_threads = threads.start(); 
	assert(m_amount_of_neurons% number_of_threads == 0);
	size_t split = m_amount_of_neurons/ number_of_threads; 
	return split; 
}

void layer::forward_layer(value_array& in)
{
	auto first = std::chrono::high_resolution_clock::now(); 
	auto function = [](std::vector<neuron>& neural){
		std::vector<value> temp;
		std::for_each(neural.begin(), neural.end(),[&](neuron& n){
		n.forward(); 
			temp.push_back(std::move(n.neuron_output())); 
		});
		return temp; };
	auto second = std::chrono::high_resolution_clock::now(); 
	auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(second - first);
	std::cout << diff.count() << std::flush; 
	std::vector<std::future<std::vector<value>>> return_val; 
	std::vector<std::vector<value>> result;  

	for(std::vector<neuron>& vec: m_neurons_fast ){
		for(neuron& n: vec)
			n.set_input(in);
		return_val.push_back(std::move(threads.enqueue((function), std::ref(vec))));
	}

	for(std::future<std::vector<value>>& x: return_val)
		result.push_back(std::move(x.get()));
	std::vector<value> answer; 
	answer.reserve(m_amount_of_neurons); 
	std::for_each(result.begin(), result.end(), [&](std::vector<value>& x){
			for(value& g: x)
				answer.push_back(std::move(g));
			});
	final = answer; 
	return;
}
void layer::normal_forward_layer(value_array& in)
{
	std::vector<value> temp; 
	std::for_each(m_neurons.begin(), m_neurons.end(), [&](neuron& n)  {
			n.set_input(in);
			n.forward();
			temp.push_back(std::move(n.neuron_output())); 
			}); 
	final = temp;
	return;
}
value_array& layer::layer_output()
{
	return final;
}
MLP::MLP(std::initializer_list<int> neurons, std::initializer_list<activation> functions)

{
	
}
