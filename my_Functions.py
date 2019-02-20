def merge_n_diff(p1, p2, pk, flt_col, str_col, how="outer"):
    
    m = pd.merge(p1, p2, left_on=pk[0], right_on=pk[1], how=how)
    
    res = m[pk[0]]
    
    for col1, col2 in zip(flt_col[0],flt_col[1]):
        
        if col1 == col2: col1, col2 = col1 + "_x", col2 + "_y"
        
        res = pd.concat([res, m[[col1,col2]], pd.Series(m[col1]-m[col2], name=col1+"_"+col2)], axis=1)
    
    for col1, col2 in zip(str_col[0],str_col[1]):
        
        if col1 == col2: col1, col2 = col1 + "_x", col2 + "_y"
        
        res = pd.concat([res, m[[col1,col2]], pd.Series(m[col1]==m[col2], name=col1+"_"+col2)], axis=1)
        
    return res
