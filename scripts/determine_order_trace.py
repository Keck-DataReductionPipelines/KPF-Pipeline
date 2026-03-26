# GJG: Here's some code I wrote inside a notebook
# to manually produce an order trace for 20240405
# Cells were run out of order interactively so this
# code probably won't work without revision, but it
# will follow something close to the needed logic

# This script should port over the order trace finding
# algorithm from the old DRP and use the logic below to
# add human-readable columns for FIBER and ORDER

# We will want one order trace per KPFERA

################

#filepath = '../reference/order_trace_red.csv'
#df = pd.read_csv(filepath, index_col=0)
#df = df.drop(index=[0]).reset_index(drop=True)

#nrow, ncol = df.shape
#fiber = np.tile('SCI1 SCI2 SCI3 CAL SKY'.split(), nrow//5)

#df = df.assign(Fiber=fiber)
#df.to_csv(filepath)

################

#filepath = '../reference/order_trace_green.csv'
#df = pd.read_csv(filepath, index_col=0)

#nrow, ncol = df.shape
#order = np.repeat(np.arange(1,36), nrow//35)

#df = df.assign(Order=order)
#df.to_csv(filepath)

################

#filepath = '../reference/order_trace_red.csv'
#df = pd.read_csv(filepath, index_col=0)

#nrow, ncol = df.shape
#order = np.hstack([np.repeat(np.arange(1,33), nrow//32)[1:],33])

#df = df.assign(Order=order)
#df = df.drop(index=[159]).reset_index(drop=True)
#df
#df.to_csv(filepath)