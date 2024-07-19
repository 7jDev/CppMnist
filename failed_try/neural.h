#include "value.h"
#include <fstream>
#include <string.h>
#include <filesystem>
#include <array>
#include <initializer_list>
#include <thread>
#include <mutex>
#include <tuple> 
#include <condition_variable>
#include <queue>
#include <functional>
#ifndef THREAD_H
#define THREAD_H
template <typename T, typename T2>
class threadPool
{

	public:
		size_t start(){
		const size_t number_of_threads = std::thread::hardware_concurrency();
		for(size_t _=0; _<number_of_threads; _++){
			threads.emplace_back(std::thread(&threadPool::threadLoop, this));
		}
		return number_of_threads;
		}
		std::vector<T2>& return_output()
		{
		{
		std::unique_lock<std::mutex> lock(queue_mutex);
		terminate = true;
		}
		mutex_condition.notify_all();
		for(std::thread& active_threads: threads ){
			if(active_threads.joinable())
				active_threads.join();
		}
		std::unique_lock<std::mutex> lock(answer_mutex);	
		return answers;
		}
		void stop()
		{
		{
		std::unique_lock<std::mutex> lock(queue_mutex);
		terminate = true;
		}
		mutex_condition.notify_all();
		for(std::thread& active_threads: threads )
			active_threads.join();
		}
		bool busy()
		{
		bool termination;
		{
		std::unique_lock<std::mutex> lock(queue_mutex);
		termination = !jobs.empty(); 
		}
		return termination; 
		} 
		void jobQueue(std::function<T2(T&)>& job, T& inputs){
			{
			std::unique_lock<std::mutex> lock(queue_mutex);
			std::unique_lock<std::mutex> lock_input(input_mutex);
			input.push(std::ref(inputs)); 
			jobs.push(std::move(job));
			}
			mutex_condition.notify_one(); 
			return; 
		}
	private:
		void threadLoop()
		{
		while(true)
		{
			std::function<T2(T&)> job; 
			{
			std::unique_lock<std::mutex> lock(queue_mutex);
			mutex_condition.wait(lock, [this] {return !jobs.empty() || terminate;});
			if(terminate)
				return;
			job = jobs.front();
			jobs.pop();
			}
			{
			std::unique_lock<std::mutex> lock(input_mutex);
			std::unique_lock<std::mutex> answer_lock(answer_mutex);
			std::vector<value> add = job(input.front().get());
			input.pop();
			answers.push_back(std::move(add));
			}
		}
		}
		std::vector<std::thread> threads;  	
		std::condition_variable mutex_condition; 
		std::queue<std::function<T2(T&)>> jobs;
		std::mutex queue_mutex;
		std::mutex answer_mutex;
		std::mutex input_mutex; 
		std::queue<std::reference_wrapper<T>> input; 
		std::vector<T2> answers; 
		bool terminate;
};
#endif
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
	friend std::ostream& operator<<(std::ostream& os, neuron& input);
	neuron();
	neuron(size_t input,activations act); 
	activations return_activation(); 
	void n_output(value_array& input);
	value& return_output();
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
	        value_array soft_output;
		std::vector<value_array> input_normal; 
		static threadPool<std::vector<value_array>, std::vector<value>> pool;
		layer* next;
		void set_input(value_array& input); 
		std::vector<std::tuple<std::vector<value_array>,std::vector<std::reference_wrapper<neuron>>>> split_neurons(value_array& inputs);
		std::vector<value> apply(std::vector<value_array>& input,std::vector<std::reference_wrapper<neuron>>& neuro); //std::tuple<std::vector<value_array>, std::vector<std::reference>>
	public:
		layer();
		void set_next(layer * next); 
		layer(size_t amount, size_t input, activations act);
		void normal_output(value_array& input);
		void l_output(value_array& input);
		activations return_activation();
		void softmax();
		value cross_entropy(int i);
		layer * get_next();
		layer * return_this();
	       value_array& return_l_output();	
};
#endif
#ifndef MLP_H
#define MLP_H
class mlp 
{
	private:
		std::vector<int> layer_neurons; 
		std::vector<layer>layers;
		value_array current_data_input; 
		std::vector<activations> functions;
		std::vector<std::string> current_directory; 
		std::string current_path;
		value loss; 
		int current_class; 
		int input; 	
	public:

	mlp();
	mlp(int input_size,  std::initializer_list<int> temp, std::initializer_list<activations> activation_func);
	std::vector<std::string> delimit(const std::string& input, const char stop);
	char get_current_class();
	std::vector<std::string>& start(std::string& path, const char n_class);
	std::vector<double> get_file( std::string& path, const char n_class); 
	value_array& predict_helper(layer & current, value_array& input);
	value_array& predict(value_array& input);
	void one_epoch_h(value_array& input);
	value& one_epoch();
	void calculate_gradients();
	value_array& get_data(std::string& path);
};

#endif

