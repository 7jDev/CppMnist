#include "value.h"
value::value(double data): m_data(data), m_gradient(0){}
value* value::return_ptr(){
	m_alloc = true; 
	value * temp = new value; 
	*temp = *this; 
	return temp; 
}
