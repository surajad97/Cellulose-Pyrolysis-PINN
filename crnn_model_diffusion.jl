ckpt_path2 = string("./results_catalyst/5s4r/checkpoint")
@load string(ckpt_path2, "/mymodel.bson") p #opt l_loss_train l_loss_val list_grad iter
#p[9] = -2
p_cutoff = 0

function display_p(p)
    w_in, w_b, w_out = p2vec(p)
    println("\n species (column) reaction (row)")
    println("w_in | w_cat_in | Ea | b | lnA | w_out | w_cat_out")
    show(stdout, "text/plain", round.(hcat(w_in', w_b, w_out'), digits = 2))
    # println("\n w_out")
    # show(stdout, "text/plain", round.(w_out', digits=3))
    println("\n")
end

display_p(p)

w_in, w_b, w_out = p2vec(p);

nr_q = 1
ns_q = 6
nc_q = 1
nss_q = ns_q - nc_q

np_q = nr_q * (ns_q + nc_q + 3) + 1
q = randn(Float64, np_q) .* 5.e-2;
q[1:nr_q] .+= 0.8;  # w_lnA
#p[nr*(nss+1)+1:nr*(ns+1)].*= 10           #w_cat_in
q[nr_q*(ns_q+1)+1:nr_q*(ns_q+nc_q+1)] .= 0         #w_cat_outp
q[nr_q*(ns_q+nc_q+1)+1:nr_q*(ns_q+nc_q+2)] .+= 0.8;  # w_Ea
q[nr_q*(ns_q+nc_q+2)+1:nr_q*(ns_q+nc_q+3)] .+= 0.1;  # w_b
q[end] = 0.1;  # slope

function q2vec(q)
    slope_q = q[end] .* 1.e1
    w_b_q = q[1:nr_q] .* (slope_q * 10.0)
    w_b_q = clamp.(w_b_q, 0, 50)

    w_out_q = reshape(q[nr_q+1:nr_q*(nss_q+1)], nss_q, nr_q)
    @. w_out_q[1, :] = clamp(w_out_q[1, :], -3.0, 0.0)
    @. w_out_q[end, :] = clamp(abs(w_out_q[end, :]), 0.0, 3.0)

    if q_cutoff > 0.0
       w_out_q[findall(abs.(w_out_q) .< q_cutoff)] .= 0.0
    end

    w_out_q[nss_q-1:nss_q-1, :] .=
        -sum(w_out_q[1:nss_q-2, :], dims = 1) .- sum(w_out_q[nss_q:nss_q, :], dims = 1)

    w_cat_in_q = q[nr_q*(nss_q+1)+1:nr_q*(ns_q+1)]
    w_cat_out_q = q[nr_q*(ns_q+1)+1:nr_q*(ns_q+nc_q+1)]*0
    
    #w_cat_out = - sign.(w_cat_in) .* abs.(w_cat_out);
    w_cat_in_q = abs.(w_cat_in_q)
    
    if q_cutoff > 0.0
        w_cat_in_q[findall(abs.(w_cat_in_q) .< q_cutoff)] .= 0.0
    end
    
    w_in_Ea_q = abs.(q[nr_q*(ns_q+nc_q+1)+1:nr_q*(ns_q+nc_q+2)].* (slope_q * 100.0))
    w_in_Ea_q = clamp.(w_in_Ea_q, 0.0, 300.0)

    w_in_b_q = abs.(q[nr_q*(ns_q+nc_q+2)+1:nr_q*(ns_q+nc_q+3)])

    w_in_q = vcat(clamp.(-w_out_q, 0.0, 3.0), w_cat_in_q', w_in_Ea_q', w_in_b_q')
    w_out_q = vcat(w_out_q, w_cat_out_q')
    
    w_in_q = hcat(w_in, w_in_q)
    w_b_q = vcat(w_b, w_b_q)
    w_out_q = hcat(w_out, w_out_q)
    
    return w_in_q, w_b_q, w_out_q
end