#include "value.h"
#include <thread>
#include <functional>
#include <queue>
#include <future>
#include <condition_variable>
#ifndef THREADP_H
#define THREADP_H
class ThreadPool{
	public:
		size_t start(){
		started = true; 
		size_t number_of_threads = std::thread::hardware_concurrency(); 
		for(size_t ii =0; ii < number_of_threads; ii++)
			threads.emplace_back(std::thread(&ThreadPool::threadLoop, this)); 
		return number_of_threads;
		}
		bool is_started(){
		return started; 
		}
		template<typename T, typename... Args>
		auto enqueue(T&& function, Args&&... arg) -> std::future<decltype(function(arg...))>
		{
			typedef decltype(function(arg...)) return_type;
			auto task = std::make_shared<std::packaged_task<return_type(void)>>(std::bind(std::forward<T>(function), std::forward<Args>(arg)...));
			{
			std::unique_lock<std::mutex> lock(queue_mutex);
			jobs.emplace([task](void){(*task)();});
			}
			cv.notify_one(); 
			return task->get_future();
		}
		~ThreadPool()
		{
		{
		std::unique_lock<std::mutex> lock(queue_mutex);
		terminate = true; 
		}
		cv.notify_all(); 
		for(std::thread& x: threads)
			x.join();
		}
	private:
		void threadLoop()
		{
		while(true){
			std::function<void()> job; 
		{
		std::unique_lock<std::mutex> lock(queue_mutex);
		cv.wait(lock,[this](void){return !jobs.empty() || terminate;});
		if(terminate)
			return;
		job  = std::move(jobs.front()); 
		jobs.pop(); 
		}
		job(); 
		}
		}
		bool started = false; 
		std::vector<std::thread> threads; 
		std::queue<std::function<void()>> jobs;
		std::mutex queue_mutex; 
		std::condition_variable cv;
		bool terminate; 
};
#endif 
#ifndef NEURON_H
#define NEURON_H
enum activation{
	TANH, 
	ELU, 
	NONE 
};
class neuron{
	public:
		neuron(); 
		neuron(size_t input_size, activation func);
		void forward(); 
		void set_input(value_array & in);
		value& neuron_output(); 
	private:
		activation function; 
		value_array weights; 
		value bias;
		value_array input; 
		value_array weights_input; 
		value sum; 
		value sumwbias; 
		value final;
}; 
#endif
#ifndef LATER_H
#define LAYER_H
class layer{
	public:
		layer();
		layer(size_t input_size, size_t amount_of_neurons, activation func, bool threadable);
		void normal_forward_layer(value_array& in); 
		void forward_layer(value_array& in);
		value_array& layer_output();
	private:
		size_t split_up();
		std::vector<neuron> m_neurons;
		std::vector<std::vector<neuron>> m_neurons_fast; 
		value_array final;
		size_t m_amount_of_neurons ;
		activation function;
		static ThreadPool threads;   
};
#endif 
#ifndef NEURAL_NET_H
#define NERUAL_NET_H
class MLP
{
	public:
		MLP(std::initializer_list<int> neurons, 
				std::initializer_list<activation> functions); 	
	private: 
		std::vector<layer> layers; 
};
#endif
