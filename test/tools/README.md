How to generate test-data file for WPT tests?

Step 1: Please prepare resources JSON file which includes those tests
to test each operator of WebNN API without specified inputs and outputs
data.

Step 2: Implement generate test-data scripts

Step 3: Execute command for generating test-data files, take an example
for reduceL1 op

```shell
node gen-reduce.js resources\\reduce_l1.json
```

then, you can find two generated folders named 'test-data' and
'test-data-wpt'. There're raw test data as being
./test-data/reduce_l1-data.json,
and raw WPT test-data file as being ./test-data-wpt/reduce_l1.json.


You can manually modify some test data in
./test-data/reduce_l1-data.json,
then execute Step 3, to update ./test-data-wpt/reduce_l1.json.