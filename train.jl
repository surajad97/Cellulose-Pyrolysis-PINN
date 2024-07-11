include("header.jl")
include("load_data.jl")
include("crnn_model.jl")
include("plotting.jl")

epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float64, n_exp);
grad_norm = zeros(Float64, n_exp);
for epoch in epochs
    global p
    for i_exp in randperm(n_exp)
        if i_exp in l_val
            continue
        end
        grad = ForwardDiff.gradient(x -> loss_neuralode(x, i_exp), p)
        grad_norm[i_exp] = norm(grad, 2)
        if grad_norm[i_exp] > grad_max
            grad = grad ./ grad_norm[i_exp] .* grad_max
        end
        update!(opt, p, grad)
    end
    for i_exp = 1:n_exp
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_train = mean(loss_epoch[l_train])
    loss_val = mean(loss_epoch[l_val])
    grad_mean = mean(grad_norm[l_train])
    set_description(
        epochs,
        string(
            @sprintf(
                "Loss train: %.2e val: %.2e grad: %.2e lr: %.1e",
                loss_train,
                loss_val,
                grad_mean,
                opt[1].eta
                #opt.eta
            )
        ),
    )
    cb(p, loss_train, loss_val, grad_mean)
end

conf["loss_train"] = minimum(l_loss_train)
conf["loss_val"] = minimum(l_loss_val)
YAML.write_file(config_path, conf)

expr_name = "5s6r-03"
fig_path = string("./results_catalyst/", expr_name, "/figs")
ckpt_path = string("./results_catalyst/", expr_name, "/checkpoint")
@load string(ckpt_path, "/mymodel.bson") p #opt l_loss_train l_loss_val list_grad iter

#p[9] = -2
p_cutoff = 0.0

display_p(p)

for i_exp = 1:n_exp
    cbi(p, i_exp)
end

loss_epoch = zeros(Float64, n_exp);
for i_exp = 1:n_exp
    loss_epoch[i_exp] = loss_neuralode(p, i_exp)
end
loss_train = mean(loss_epoch[l_train])
loss_val = mean(loss_epoch[l_val])
loss_train, loss_val