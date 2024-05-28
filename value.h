#include <vector>
#include <ostream>
#include <cmath>
#include <algorithm>
#include "queue.h"
#include <random>
#ifndef VALUE_H
#define VALUE_H
class value {
private:
	double m_gradient;
       	double m_data;
	std::vector<value*> m_parents;
	Node<value*> m_node;
	value* m_ptr = this;
public:
	friend std::ostream& operator<<(std::ostream& os, value& input);
	value();
	value(const value& other) = delete;
	value(value& other) =delete;
	value(value&& other);
	value(double data);
	void random_init();
	value operator+(value& x); 
	value operator*(value& x);
	value tanh();
	void calculate_gradients();
};
 
#endif
#ifndef VALUE_ARRAY_H
#define VALUE_ARRAY_H
class value_array{
	private:
	std::vector<value> m_values;
	public:
	value_array();
};
#endif
