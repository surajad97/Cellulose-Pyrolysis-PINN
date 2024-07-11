function load_exp(filename)
    exp_data = readdlm(filename)  # [t, T, m]
    index = indexin(unique(exp_data[:, 1]), exp_data[:, 1])
    exp_data = exp_data[index, :]
    exp_data[:, 3] = exp_data[:, 3] / maximum(exp_data[:, 3])
    return exp_data
end

l_exp_data = [];
l_exp_info = zeros(Float64, length(l_exp), 3);
for (i_exp, value) in enumerate(l_exp)
    #filename = string("Exp_data_full_trimmed/expdata_no", string(value), ".txt")
    filename = string("exp_data/expdata_no", string(value), ".txt")

    exp_data = Float64.(load_exp(filename))

    push!(l_exp_data, exp_data)
    l_exp_info[i_exp, 1] = exp_data[1, 2] # initial temperature, K
end
#l_exp_info[:, 2] = readdlm("Exp_data_full_trimmed/beta.txt")[l_exp];
#l_exp_info[:, 3] = (readdlm("Exp_data_full_trimmed/cata_conc.txt")[l_exp]);
l_exp_info[:, 2] = readdlm("exp_data/beta.txt")[l_exp];
l_exp_info[:, 3] = (readdlm("exp_data/cata_conc.txt")[l_exp]);