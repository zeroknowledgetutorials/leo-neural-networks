neurons_per_layer = [2, 2, 1] # specifies NN architecture
scaling_factor = 1 # specifies scaling factor for fixed point numbers
integer_type = "u32" # specifies used integer type
 
if(len(neurons_per_layer) < 2 or min(neurons_per_layer) < 1):
   print("error, invalid input")
 
str_list_main = []
str_list_inputs = []
 
str_main="function main("
 
str_inputs = ""
 
str_list_inputs.append("[main]\n")
 
for i in range(neurons_per_layer[0]):
   str_main += "w0" + str(i)+": " + integer_type + ", b0" + str(i) + ": " + integer_type + ", "
   str_inputs += "w0" + str(i) + ": " + integer_type + " = 0;\n"
   str_inputs += "b0" + str(i) + ": " + integer_type + " = 0;\n"
 
str_list_inputs.append(str_inputs)
str_inputs = ""
 
for i in range(1, len(neurons_per_layer)): # current layer
   for j in range(neurons_per_layer[i-1]): # neuron of previous layer
       for k in range(neurons_per_layer[i]): # neuron of current layer
           str_main += "w" + str(i) + str(j) + str(k) + ": " + integer_type + ", "
           str_inputs += "w" + str(i) + str(j) + str(k) + ": " + integer_type + " = 0;\n"
       str_main += "b" + str(i) + str(j) + ": " + integer_type + ", "
       str_inputs += "b" + str(i) + str(j) + ": " + integer_type + " = 0;\n"
      
for i in range(neurons_per_layer[0]):
   str_main += "input"+str(i)+": " + integer_type + ", "
   str_inputs += "input"+str(i)+": " + integer_type + " = 0;\n"
 
str_main = str_main[:-2]
str_list_inputs.append(str_inputs)
 
str_inputs = "[registers]\n"
 
str_main += ") -> [" + integer_type + "; " + str(neurons_per_layer[-1]) + "] {\n"
 
str_list_main.append(str_main)
 
line = ""
 
for i in range(neurons_per_layer[0]): # input layer
   line += "let neuron0"+str(i) + ": " + integer_type + " = w0" + str(i) + " * input" + str(i) + " / " + str(2**scaling_factor) + " + b0" + str(i) + ";\n"
 
for layer in range(1, len(neurons_per_layer)): # other layers
   for i in range(neurons_per_layer[layer]):
       line_start = "let neuron" + str(layer) + str(i) + ": " + integer_type + " = rectified_linear_activation("
       for j in range(neurons_per_layer[layer-1]):
           line_start += "neuron" + str(layer-1) + str(j) + " * w" + str(layer) + str(j) + str(i) + " / " + str(2**scaling_factor) + " + "
      
       line_start += "b" + str(layer) + str(i) + ");\n"
       line += line_start
      
str_list_main.append(line)
 
line = "return ["
str_inputs += "r0: [" + integer_type + "; " + str(neurons_per_layer[-1]) + "] = ["
for i in range(neurons_per_layer[-1]):
   line += "neuron" + str(len(neurons_per_layer)-1) + str(i) + ", "
   str_inputs += "0, "
str_inputs = str_inputs[:-2] + "];\n"
 
line = line[:-2]
line += "];}\n\n"
str_list_main.append(line)
str_list_inputs.append(str_inputs)
 
str_list_main.append("function rectified_linear_activation(x: u32) -> u32 {\n")
str_list_main.append("let result: u32 = 0;\n")
str_list_main.append("if x > 0 {\n")
str_list_main.append("result = x;\n")
str_list_main.append("}\n")
str_list_main.append("return result;\n")
str_list_main.append("}")
 
with open("main.leo", "w+") as file:
   file.writelines(str_list_main)
 
with open("project.in", "w+") as file:
   file.writelines(str_list_inputs)