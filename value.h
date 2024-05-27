#include <vector>
#ifndef VALUE_H
#define VALUE_H
class value {
private:
	double m_gradient;
       	double m_data;
	std::vector<value*> m_parents;
	bool m_alloc = false; 
public:
	value();
	value(double data);
 	static value * return_ptr();
};
#endif 
