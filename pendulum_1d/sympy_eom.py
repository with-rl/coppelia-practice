import sympy as sy


def main():
    theta1 = sy.symbols("theta1", real=True)
    l1 = sy.symbols("l1", real=True)

    #
    # 1. Homogeneous Matrix
    #
    H_01 = sy.Matrix(
        [
            [sy.cos(theta1), -sy.sin(theta1), 0],
            [sy.sin(theta1), sy.cos(theta1), 0],
            [0, 0, 1],
        ]
    )
    print("*" * 20, "Homogeneous Matrix", "*" * 20)
    print(f"H_01: {H_01}")
    print()

    #
    # 2. Jacobian
    #
    q = [theta1]

    C_1 = H_01 * sy.Matrix([l1, 0, 1])
    C_1 = sy.Matrix([C_1[0], C_1[1]])
    J_1 = C_1.jacobian(q)

    print("*" * 20, "Jacobian", "*" * 20)
    print(f"C_1: {C_1}")
    print(f"J_1: {J_1}")
    print()


if __name__ == "__main__":
    main()
