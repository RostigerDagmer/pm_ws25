import pm4py
from pm4py.objects.bpmn.obj import BPMN


def create_manual_bpmn_with_swimlanes():
    # Create a new BPMN diagram with collaboration
    collaboration_id = "collab_001"
    bpmn_diagram = BPMN(process_id=collaboration_id, name="Process with Swimlanes")

    # Create collaboration
    collaboration = BPMN.Collaboration(id=collaboration_id, name="Main Collaboration")
    bpmn_diagram.add_node(collaboration)

    # Create two participants (swimlanes)
    main_process_id = "proc_main_001"
    sub_process_id = "proc_sub_001"

    main_participant = BPMN.Participant(
        id="part_main_001",
        name="Main Department",
        process_ref=main_process_id
    )
    sub_participant = BPMN.Participant(
        id="part_sub_001",
        name="Sub Department",
        process_ref=sub_process_id
    )

    bpmn_diagram.add_node(main_participant)
    bpmn_diagram.add_node(sub_participant)

    # Main process nodes
    main_start = BPMN.NormalStartEvent(name="Start Process", process=main_process_id)
    main_task1 = BPMN.Task(name="Perform Task 1", process=main_process_id)
    main_task2 = BPMN.Task(name="Perform Task 2", process=main_process_id)
    main_end = BPMN.NormalEndEvent(name="End Process", process=main_process_id)

    # Add main process nodes
    bpmn_diagram.add_node(main_start)
    bpmn_diagram.add_node(main_task1)
    bpmn_diagram.add_node(main_task2)
    bpmn_diagram.add_node(main_end)

    # Sub process nodes
    sub_start = BPMN.NormalStartEvent(name="Sub Start", process=sub_process_id)
    sub_task = BPMN.Task(name="Sub Task", process=sub_process_id)
    sub_end = BPMN.NormalEndEvent(name="Sub End", process=sub_process_id)

    # Add sub process nodes
    bpmn_diagram.add_node(sub_start)
    bpmn_diagram.add_node(sub_task)
    bpmn_diagram.add_node(sub_end)

    # Create sequence flows within each swimlane
    main_flow1 = BPMN.SequenceFlow(main_start, main_task1, name="Flow 1", process=main_process_id)
    main_flow2 = BPMN.SequenceFlow(main_task1, main_task2, name="Flow 2", process=main_process_id)
    main_flow3 = BPMN.SequenceFlow(main_task2, main_end, name="Flow 3", process=main_process_id)

    sub_flow1 = BPMN.SequenceFlow(sub_start, sub_task, name="Sub Flow 1", process=sub_process_id)
    sub_flow2 = BPMN.SequenceFlow(sub_task, sub_end, name="Sub Flow 2", process=sub_process_id)

    # Add sequence flows
    bpmn_diagram.add_flow(main_flow1)
    bpmn_diagram.add_flow(main_flow2)
    bpmn_diagram.add_flow(main_flow3)
    bpmn_diagram.add_flow(sub_flow1)
    bpmn_diagram.add_flow(sub_flow2)

    # Add a message flow between swimlanes
    message_flow = BPMN.MessageFlow(
        source=main_task2,
        target=sub_start,
        name="Request Sub Process",
        process=collaboration_id
    )
    bpmn_diagram.add_flow(message_flow)

    # Set some layout information for visualization
    # Main swimlane (vertically stacked)
    main_participant.set_x(0)
    main_participant.set_y(0)
    main_participant.set_width(600)
    main_participant.set_height(200)

    main_start.set_x(50)
    main_start.set_y(80)
    main_task1.set_x(150)
    main_task1.set_y(80)
    main_task2.set_x(250)
    main_task2.set_y(80)
    main_end.set_x(350)
    main_end.set_y(80)

    # Sub swimlane (below main)
    sub_participant.set_x(0)
    sub_participant.set_y(200)
    sub_participant.set_width(600)
    sub_participant.set_height(200)

    sub_start.set_x(50)
    sub_start.set_y(280)
    sub_task.set_x(150)
    sub_task.set_y(280)
    sub_end.set_x(250)
    sub_end.set_y(280)

    return bpmn_diagram


def merge_bpmn_with_swimlanes(main_bpmn, bpmn_to_merge):
    """
    Merge another BPMN into the main BPMN diagram while maintaining swimlanes
    """
    # Find the collaboration
    collaboration = next(n for n in main_bpmn.get_nodes() if isinstance(n, BPMN.Collaboration))

    # Create a new participant for the merged process
    new_process_id = bpmn_to_merge.get_process_id()
    new_participant = BPMN.Participant(
        id=f"part_{new_process_id}",
        name=f"Merged Process {new_process_id}",
        process_ref=new_process_id
    )
    main_bpmn.add_node(new_participant)

    # Set layout for new swimlane (below existing ones)
    participants = [n for n in main_bpmn.get_nodes() if isinstance(n, BPMN.Participant)]
    max_y = max(p.get_y() + p.get_height() for p in participants[:-1])
    new_participant.set_x(0)
    new_participant.set_y(max_y)
    new_participant.set_width(600)
    new_participant.set_height(200)

    # Add nodes with offset
    id_mapping = {}
    y_offset = max_y - list(bpmn_to_merge.get_nodes())[0].get_y() if bpmn_to_merge.get_nodes() else 0

    for node in bpmn_to_merge.get_nodes():
        new_node = type(node)(
            id=node.get_id(),
            name=node.get_name(),
            process=new_process_id
        )
        new_node.set_x(node.get_x())
        new_node.set_y(node.get_y() + y_offset)
        new_node.set_width(node.get_width())
        new_node.set_height(node.get_height())

        main_bpmn.add_node(new_node)
        id_mapping[node.get_id()] = new_node

    # Add flows
    for flow in bpmn_to_merge.get_flows():
        source = id_mapping[flow.get_source().get_id()]
        target = id_mapping[flow.get_target().get_id()]

        new_flow = type(flow)(
            source=source,
            target=target,
            id=flow.get_id(),
            name=flow.get_name(),
            process=new_process_id
        )

        for waypoint in flow.get_waypoints():
            new_flow.add_waypoint((waypoint[0], waypoint[1] + y_offset))

        main_bpmn.add_flow(new_flow)

    return main_bpmn


def print_bpmn_summary(bpmn, name):
    print(f"\n{name}:")
    print(f"Process ID: {bpmn.get_process_id()}")
    print(f"Process Name: {bpmn.get_name()}")

    participants = [n for n in bpmn.get_nodes() if isinstance(n, BPMN.Participant)]
    print(f"Number of swimlanes: {len(participants)}")
    print(f"Total nodes: {len(bpmn.get_nodes())}")
    print(f"Total flows: {len(bpmn.get_flows())}")

    for participant in participants:
        print(f"\nSwimlane: {participant.get_name()} (Process: {participant.process_ref})")
        print(f"Position: ({participant.get_x()}, {participant.get_y()})")
        print(f"Size: {participant.get_width()}x{participant.get_height()}")
        print("Nodes:")
        nodes = [n for n in bpmn.get_nodes() if n.get_process() == participant.process_ref]
        for node in nodes:
            if not isinstance(node, BPMN.Participant):
                print(f"  - {node.get_name()} ({type(node).__name__}) at ({node.get_x()}, {node.get_y()})")
        print("Flows:")
        flows = [f for f in bpmn.get_flows() if f.get_process() == participant.process_ref]
        for flow in flows:
            print(f"  - {flow.get_source().get_name()} -> {flow.get_target().get_name()}")

    # Print message flows
    message_flows = [f for f in bpmn.get_flows() if isinstance(f, BPMN.MessageFlow)]
    if message_flows:
        print("\nMessage Flows:")
        for flow in message_flows:
            print(f"  - {flow.get_source().get_name()} -> {flow.get_target().get_name()}")


def execute_script():
    # Create the main BPMN with swimlanes
    main_bpmn = create_manual_bpmn_with_swimlanes()

    # Create a simple BPMN to merge
    simple_bpmn = BPMN(process_id="proc_simple_001", name="Simple Process")
    s_start = BPMN.NormalStartEvent(name="Simple Start")
    s_task = BPMN.Task(name="Simple Task")
    s_end = BPMN.NormalEndEvent(name="Simple End")
    simple_bpmn.add_node(s_start)
    simple_bpmn.add_node(s_task)
    simple_bpmn.add_node(s_end)
    simple_bpmn.add_flow(BPMN.SequenceFlow(s_start, s_task))
    simple_bpmn.add_flow(BPMN.SequenceFlow(s_task, s_end))

    # Print original diagram
    print_bpmn_summary(main_bpmn, "Original BPMN with Swimlanes")

    pm4py.view_bpmn(main_bpmn, format="svg")
    pm4py.view_bpmn(simple_bpmn, format="svg")

    # Merge the simple BPMN
    merged_bpmn = merge_bpmn_with_swimlanes(main_bpmn, simple_bpmn)

    # Print the merged result
    print_bpmn_summary(merged_bpmn, "Merged BPMN with Swimlanes")

    pm4py.view_bpmn(merged_bpmn, format="svg")


# Main execution
if __name__ == "__main__":
    execute_script()
