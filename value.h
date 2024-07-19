#include <vector>
#include <ostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <queue>
#include <cassert>
#ifndef VALUE_H
#define VALUE_H
class value {
private:
	double m_gradient;
       	double m_data;
	bool m_grad = false;
	std::vector<std::reference_wrapper<value>>m_parents;
	value* m_ptr = this;
public:
	friend std::ostream& operator<<(std::ostream& os, value& input);
	value();
	value(const value& other)= delete;
	value(value& other) = delete;
	value(value&& other);
	value& operator=(value&& other);
	value& operator=(const value& other) = default;
	bool operator==(value& other);
	value& operator+=(value& other);
	void push_back(value& item);
	value return_copy();
	value(double data);
	void random_init();
	void requires_grad();
	void change_gradient(double x);
	double return_data();
	value operator+(value& x); 
	value operator*(value& x);
	value tanh();
	value elu();
	void calculate_gradients();
	void learn(double lr);
};
 
#endif
#ifndef VALUE_ARRAY_H
#define VALUE_ARRAY_H
class value_array{
        private:
        std::vector<value> m_values;
        size_t len;
	double clip(double input, double lower, double higher); 
	public:
        value_array();
	size_t size();
	friend std::ostream& operator<<(std::ostream& os, value_array& input); 
        value_array(size_t length);
	value_array(std::vector<double>& items);
	value_array(const value_array& other) = delete;
       	value_array(value_array& other) = delete; 
	value_array return_copy();
	value_array(value_array&& other);
       	value cross_entropy(int i);	
	value_array& operator=(value_array&& other);
	value_array& operator=(std::vector<value>& temp);
	value_array& operator=(const value_array& other) = delete;
	value& operator[](const size_t i);
	void requires_grad();
	void random_init();
	value_array softmax();
	value_array operator+(value_array& other);
        value_array operator*(value_array& other);
        value sum();
	std::vector<value>& return_vector();
};
#endif
