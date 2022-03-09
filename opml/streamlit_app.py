# import sympy
# import streamlit as st
#
# from function_parser.sympy_parser import Text2Sympy

# st.write('# Hello world')
# function = st.text_input('Enter the function here', 'x**2 + x + 1')
# function = Text2Sympy.parse_func(function)
# function = function.subs({sympy.symbols('e'): sympy.exp(1)})
# st.write('> $$ f(x) = ', rf'\ {sympy.latex(function)}$$')
# value = st.text_input('Enter the point here', '2')
# value = Text2Sympy.parse_func(value)
# st.write('> $$ f(x) = ', rf'\ {sympy.latex(function.subs({list(function.free_symbols)[0]: value}))}$$')


# if __name__ == '__main__':
#     print(Text2Sympy)
