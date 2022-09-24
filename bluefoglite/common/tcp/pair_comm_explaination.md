# Send and Receive Communication Timeline Explanation

### Old Passive Eventloop Style 
In the the old style, the pair class contains three key components:
1.  `_pending_send: Deque[Envelope]`
2.  `_pending_recv: Deque[Envelope]`
3.  `_event_loop` and the corresponding `handleConnected` callback functions.

The workflow is in the main thread create envolope of send/recv request and push the envolope to the queue. The work in main thread is done since the rest work is done in the communication thread entirely. In the communication thread, it waits until `event_loop` trigged that `selectors.EVENT_WRITE` is ready. In the `handleConnected` function, it called the `self.sock`'s send/recv function to receive the header by the fixed bytes. Then use the information in the header to send/receive the full bytes. When all of above operations is done, the `handleCompletion` function associated in the `buf` will be called, which will let the main thread know this send/recv operation is finished.


### New half-active/passive two-phase eventloop style.

The main difference between this new style versus the old style is we send out the header alone first and wait until the another side checked and stated it is ready, then we send the full informations.

To enable that, we need to extend status variables in the pair. Assuming the send/recv within one pair
is always in order, we have
1. A single `remote_ready_to_recv` envolope (w/o buff), if it is not none means it is waiting to be recv next. 
2. A list of `remote_ready_to_send` envolopes (w/o buff), if length > 0  means there is request waiting other side
   sending
3. A list of `pending_send` envolope (w/ buff), if none we need to find the correpoding one
   in the list of recv envolope
4. A list of `pending_recv` envolope (w/ buff), if none we should send the singleton send envolop immediately.
5. A single `read_to_recv` envolope (w/o buff)

**Send Function call (main thread)**
1. Create the envolope of send and check if there is wait for receiving request.
2. Depending if there is `remote_ready_to_recv` 
   1. If no, first pushed the send envolope into the `pending_send` queue then send out the potential sending message (i.e. header only w/o buff) immediately to the remote.
   2. If yes, send out the ready to send message (i.e. header only w/o buff)  immediately then send out  the full message, i.e. full message w/ buff.
   3. (Optional?) If yes, should we check the signature of the receiving one matched the sending one? If not matched, throw an error since we assume within the pair the communication is in order?

**Recv Function call (main thread)**
1. Create the envolope of recv and check if there is wait for sending request.
2. No matter if there is `remote_ready_to_send` or not, first pushed the send envolope into the `pending_recv` then send the `ready_to_recv` message (i.e. header only w/o buff) to other side if `read_to_recv` is None. (if not none, skip this chance).

**Send Function call (communication thread):**
1. Wait the selectors.EVENT_WRITE popped then tested based on status:
    1. `remote_ready_to_recv` variable is None, it means the other side is not prepared and we should just skip.
    2. `remote_ready_to_recv` variable is there, take the first element of `pending_send` out and send it.
    3. (Optional?) We can check the signature of remote_ready_to_recv and the first element of pending_send
   
**Recv Function call (communication thread)**
The key thing here is we need to know whether we are prepared to receive the header only or the full message and if it is full message, which buff to use for recieving. Note the sending side will only send the full message when a) the receive side send out the `ready_to_recv` message and b) receive side received the `read_to_send` message, i.e. `remote_ready_to_send` has length > 0.
1. Wait the selectors.EVENT_READ popped then tested based on status:
   1. If above two conditions are not satisified, 