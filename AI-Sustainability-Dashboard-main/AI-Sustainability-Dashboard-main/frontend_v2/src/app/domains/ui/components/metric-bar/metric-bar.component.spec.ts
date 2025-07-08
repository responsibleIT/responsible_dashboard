import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MetricBarComponent } from './metric-bar.component';

describe('MetricBarComponent', () => {
  let component: MetricBarComponent;
  let fixture: ComponentFixture<MetricBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MetricBarComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MetricBarComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
