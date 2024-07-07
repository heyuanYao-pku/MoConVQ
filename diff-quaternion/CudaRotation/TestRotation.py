import sympy

def calc_quat_from_matrix():
	x1, x2, x3, y1, y2, y3, z1, z2, z3 = sympy.symbols("x1, x2, x3, y1, y2, y3, z1, z2, z3")
	x_list = [x1, x2, x3, y1, y2, y3, z1, z2, z3]
	w0 = +x1+y2+z3
	w1 = +x1-y2-z3
	w2 = -x1+y2-z3
	w3 = -x1-y2+z3

	# q0_ = sympy.sqrt((1+w0)/4)
	# q1_ = (z2-y3)/(4*q0_)
	# q2_ = (x3-z1)/(4*q0_)
	# q3_ = (y1-x2)/(4*q0_)

	# q1_ = sympy.sqrt((1+w1)/4);
	# q0_ = (z2-y3)/(4*q1_);
	# q2_ = (x2+y1)/(4*q1_);
	# q3_ = (x3+z1)/(4*q1_);

	# q2_ = sympy.sqrt((1+w2)/4)
	# q0_ = (x3-z1)/(4*q2_)
	# q1_ = (x2+y1)/(4*q2_)
	# q3_ = (y3+z2)/(4*q2_)

	q3_ = sympy.sqrt((1 + w3) / 4)
	q0_ = (y1 - x2) / (4 * q3_)
	q1_ = (x3 + z1) / (4 * q3_)
	q2_ = (y3 + z2) / (4 * q3_)

	q0 = q1_;
	q1 = q2_;
	q2 = q3_;
	q3 = q0_;

	q_list = [q0, q1, q2, q3]
	for i in range(4):
		print("{")
		for j in range(9):
			print(str(sympy.simplify(q_list[i].diff(x_list[j]))) + ",", end=" ")
		print("\n},")

if __name__ == "__main__":
	calc_quat_from_matrix()
