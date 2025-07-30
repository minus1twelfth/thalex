const ws = new WebSocket('ws://localhost:8501');
let heartbeatTimeout;

const startHeartbeat = () => {
    console.log('startheartbeat');
    clearTimeout(heartbeatTimeout);
    heartbeatTimeout = setTimeout(() => {
        const status_label = document.getElementById('status_label');
        status_label.innerText = 'Timeout';
        status_label.style.color = 'red';
        console.warn('Connection timed out. Closing socket.');
        ws.close(); // or trigger reconnect logic here
    }, 3000);
};

// Checkbox click listener
document.getElementById('armed_checkbox').addEventListener('click', function() {
    const message = JSON.stringify({
        type: 'armed_checkbox',
        status: this.checked
    });
    ws.send(message);
});

document.getElementById('min_delta_input').addEventListener('change', function(event) {
    const message = JSON.stringify({
        type: 'min_delta',
        value: event.target.value
    });
    ws.send(message);
});

document.getElementById('max_delta_input').addEventListener('change', function(event) {
    const message = JSON.stringify({
        type: 'max_delta',
        value: event.target.value
    });
    ws.send(message);
});

document.getElementById('width_bid_call_input').addEventListener('change', function(event) {
    const message = JSON.stringify({
        type: 'width_bid_call',
        value: event.target.value
    });
    ws.send(message);
});

document.getElementById('width_ask_call_input').addEventListener('change', function(event) {
    const message = JSON.stringify({
        type: 'width_ask_call',
        value: event.target.value
    });
    ws.send(message);
});

document.getElementById('width_bid_put_input').addEventListener('change', function(event) {
    const message = JSON.stringify({
        type: 'width_bid_put',
        value: event.target.value
    });
    ws.send(message);
});

document.getElementById('width_ask_put_input').addEventListener('change', function(event) {
    const message = JSON.stringify({
        type: 'width_ask_put',
        value: event.target.value
    });
    ws.send(message);
});

document.getElementById('mmp_size_input').addEventListener('change', function(event) {
    const message = JSON.stringify({
        type: 'mmp_size',
        value: event.target.value
    });
    ws.send(message);
});

document.getElementById('quote_size_input').addEventListener('change', function(event) {
    const message = JSON.stringify({
        type: 'quote_size',
        value: event.target.value
    });
    ws.send(message);
});

document.getElementById('label_input').addEventListener('change', function(event) {
    const message = JSON.stringify({
        type: 'label',
        value: event.target.value
    });
    ws.send(message);
});

ws.onopen = () => {
    console.log('WebSocket connection opened.');
    const status_label = document.getElementById('status_label');
    status_label.innerText = 'Online';
    status_label.style.color = 'green';
    startHeartbeat();
};

ws.onmessage = (event) => {
    startHeartbeat();
    const data = JSON.parse(event.data);
    if ('table_data' in data) {
        updateTable(data.table_data);
    } else if ('expiry_data' in data) {
        createExpiryTable(data.expiry_data);
    } else if ('armed' in data) {
        document.getElementById('armed_checkbox').checked = data.armed;
    } else if ('open_orders' in data) {
        document.getElementById('open_orders_label').innerText = data.open_orders;
    } else if ('min_delta' in data) {
        document.getElementById('min_delta_input').value = data.min_delta;
    } else if ('max_delta' in data) {
        document.getElementById('max_delta_input').value = data.max_delta;
    } else if ('max_dte' in data) {
        document.getElementById('max_dte_input').value = data.max_dte;
    } else if ('width_bid_call' in data) {
        document.getElementById('width_bid_call_input').value = data.width_bid_call;
    } else if ('width_ask_call' in data) {
        document.getElementById('width_ask_call_input').value = data.width_ask_call;
    } else if ('width_bid_put' in data) {
        document.getElementById('width_bid_put_input').value = data.width_bid_put;
    } else if ('width_ask_put' in data) {
        document.getElementById('width_ask_put_input').value = data.width_ask_put;
    } else if ('mmp_size' in data) {
        document.getElementById('mmp_size_input').value = data.mmp_size;
    } else if ('quote_size' in data) {
        document.getElementById('quote_size_input').value = data.quote_size;
    } else if ('label' in data) {
        document.getElementById('label_input').value = data.label;
    }
};

ws.onerror = (err) => {
    const status_label = document.getElementById('status_label');
    status_label.innerText = 'Connect error';
    status_label.style.color = 'red';
    console.error('WebSocket error:', err);
};

function updateTable(data) {
    const tableHeader = document.getElementById('table-header');
    tableHeader.innerHTML = "";
    const tableBody = document.getElementById('table-body');
    tableBody.innerHTML = "";

    if (Array.isArray(data)) {
        data.forEach(addRowToTable);
    }
}

function addRowToTable(obj) {
    const tableHeader = document.getElementById('table-header');
    const tableBody = document.getElementById('table-body');

    // Create headers if not already present
    if (tableHeader.children.length === 0) {
        for (let key in obj) {
            const th = document.createElement('th');
            th.textContent = key;
            tableHeader.appendChild(th);
        }
    }

    // Create a new row
    const tr = document.createElement('tr');
    if (Object.keys(obj).length === 0) {
        const breakCell = document.createElement('td');
        const firstRow = tableBody.rows[0];
        const columnCount = firstRow ? firstRow.cells.length : 1;
        breakCell.colSpan = columnCount;
        breakCell.className = 'table-break';
        tr.appendChild(breakCell);
    } else {
        for (let key in obj) {
            const td = document.createElement('td');
            const parts = key.split('.')
            if (parts.length === 2) {
                td.style.color = parts[1];
            }
            td.textContent = obj[key];
            tr.appendChild(td);
        }
    }
    tableBody.appendChild(tr);
}

function createExpiryTable(data) {
    const headerRow = document.getElementById("expiry-table-header");
    const tableBody = document.getElementById("expiry-table-body");

    // Clear previous contents
    headerRow.innerHTML = "";
    tableBody.innerHTML = "";

    if (data.length === 0) return;

    // Build header
    const keys = Object.keys(data[0]);
    keys.forEach(key => {
        const th = document.createElement("th");
        th.textContent = key;
        headerRow.appendChild(th);
    });

    // Build rows
    data.forEach(row => {
        const tr = document.createElement("tr");

        keys.forEach(key => {
            const td = document.createElement("td");
            if (key === 'enabled') {
                const checkbox = document.createElement("input");
                checkbox.type = 'checkbox';
                checkbox.checked = row[key];
                checkbox.addEventListener("click", function(event) {
                    const message = JSON.stringify({
                        type: 'expiry_enabled',
                        exp: row[keys[0]],
                        value: event.target.checked
                    });
                    ws.send(message);
                });
                td.appendChild(checkbox);
            } else if (key.startsWith("vol offset")) {
                const input = document.createElement("input");
                input.size = 5;
                input.value = row[key];
                input.addEventListener('change', function(event) {
                    const message = JSON.stringify({
                        type: 'vol_offset',
                        exp: row[keys[0]],
                        range: key,
                        value: event.target.value
                    });
                    ws.send(message);
                });
                td.appendChild(input);
            } else {
                td.textContent = row[key];
            }
            tr.appendChild(td);
        });

        tableBody.appendChild(tr);
    });
}
